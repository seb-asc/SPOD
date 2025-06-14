import numpy as np
from scipy.sparse import diags
import yt
from tqdm import tqdm
import os
import gc
from multiprocessing import Pool, cpu_count
from functools import partial

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
GAMMA = 1.354
MACH = 1.7
FIELDS = ["density", "x_velocity", "y_velocity", "Temp"]
ROI_PARAMS = {
    'threshold': 100.0,
    'window_size': 5,
    'preset': 75,
    'offset': 1125
}
MAX_CORES = 64                     # Máximo de cores a usar
SNAPSHOTS_PER_BATCH = 5           # Procesar en lotes para evitar OOM
CHUNK_SIZE = 16                     # Tamaño de chunk para multiprocessing

# =============================================================================
# FUNCIÓN PARA PROCESAR UN SNAPSHOT INDIVIDUAL (PARALELIZABLE)
# =============================================================================
def process_snapshot(file_info):
    """Procesa un snapshot y devuelve sus ROIs con padding"""
    file, root_path, max_level, dims, target_ny = file_info
    
    try:
        ds = yt.load(os.path.join(root_path, file))
        cube = ds.covering_grid(max_level, left_edge=ds.domain_left_edge, 
                              dims=dims, fields=[("boxlib", f) for f in FIELDS])
        
        # 1. Detección dinámica de ROI (igual que en Q_hat)
        temp = cube[("boxlib", "Temp")].d[:, :, 0]
        profile = temp[0, :]
        diff_avg = np.convolve(np.diff(profile), 
                             np.ones(2*ROI_PARAMS['window_size']+1)/(2*ROI_PARAMS['window_size']+1), 
                             mode='same')
        base_candidates = np.where(diff_avg > ROI_PARAMS['threshold'])[0]
        
        base_index = dims[1] // 2 if len(base_candidates) == 0 else base_candidates[0]
        lower_bound = max(0, base_index - ROI_PARAMS['preset'])
        upper_bound = min(dims[1], base_index + ROI_PARAMS['offset'])
        
        # 2. Extraer ROI y aplicar padding
        roi_data = {}
        for field in FIELDS:
            roi = cube[("boxlib", field)].d[:, lower_bound:upper_bound, 0]
            if roi.shape[1] < target_ny:
                pad_width = ((0, 0), (0, target_ny - roi.shape[1]))
                roi = np.pad(roi, pad_width, mode='constant', constant_values=0)
            roi_data[field] = roi[:, :target_ny]  # Asegurar tamaño exacto
            
        return roi_data
    
    except Exception as e:
        print(f"Error en {file}: {str(e)}")
        return None

# =============================================================================
# FUNCIÓN PRINCIPAL (MANEJO DE BATCHES + MEMORIA)
# =============================================================================
def calculate_mean_fields_batched(snapshot_files, root_path, max_level=2):
    """Procesa en batches para evitar sobrecarga de memoria"""
    # 1. Configuración inicial
    first_ds = yt.load(os.path.join(root_path, snapshot_files[0]))
    ref = int(np.prod(first_ds.ref_factors[0:max_level]))
    dims = first_ds.domain_dimensions * ref
    nx, ny_total = dims[0], dims[1]
    
    # 2. Determinar tamaño máximo de ROI (usando un subconjunto)
    sample_files = snapshot_files[:min(100, len(snapshot_files))]
    roi_heights = []
    
    print("Calculando altura máxima de ROI...")
    for file in tqdm(sample_files):
        ds = yt.load(os.path.join(root_path, file))
        cube = ds.covering_grid(max_level, left_edge=ds.domain_left_edge, 
                              dims=dims, fields=[("boxlib", "Temp")])
        temp = cube[("boxlib", "Temp")].d[:, :, 0]
        profile = temp[0, :]
        diff_avg = np.convolve(np.diff(profile), 
                             np.ones(2*ROI_PARAMS['window_size']+1)/(2*ROI_PARAMS['window_size']+1), 
                             mode='same')
        base_candidates = np.where(diff_avg > ROI_PARAMS['threshold'])[0]
        if len(base_candidates) > 0:
            base_index = base_candidates[0]
            roi_height = min(ny_total, base_index + ROI_PARAMS['offset']) - max(0, base_index - ROI_PARAMS['preset'])
            roi_heights.append(roi_height)
    
    target_ny = max(roi_heights) if roi_heights else ROI_PARAMS['preset'] + ROI_PARAMS['offset']
    print(f"Altura de ROI objetivo: {target_ny}")
    
    # 3. Procesamiento en batches
    n_batches = (len(snapshot_files) + SNAPSHOTS_PER_BATCH - 1) // SNAPSHOTS_PER_BATCH
    accum = {field: np.zeros((nx, target_ny)) for field in FIELDS}
    
    for batch_idx in range(n_batches):
        start = batch_idx * SNAPSHOTS_PER_BATCH
        end = min((batch_idx + 1) * SNAPSHOTS_PER_BATCH, len(snapshot_files))
        batch_files = snapshot_files[start:end]
        
        print(f"\nProcesando batch {batch_idx + 1}/{n_batches} ({len(batch_files)} snapshots)")
        
        # 4. Procesamiento paralelo del batch
        n_workers = min(MAX_CORES, cpu_count(), len(batch_files))
        with Pool(n_workers) as pool:
            file_infos = [(f, root_path, max_level, dims, target_ny) for f in batch_files]
            results = list(tqdm(pool.imap(process_snapshot, file_infos, chunksize=CHUNK_SIZE),
                          total=len(batch_files), desc="Progreso"))
        
        # 5. Acumular resultados y liberar memoria
        valid_results = [r for r in results if r is not None]
        for field in FIELDS:
            batch_sum = np.sum([r[field] for r in valid_results], axis=0)
            accum[field] += batch_sum
        
        del results, valid_results
        gc.collect()
    
    # 6. Calcular promedios finales
    mean_fields = {field: accum[field] / len(snapshot_files) for field in FIELDS}
    
    # 7. Calcular volumen de celda (asumiendo malla uniforme)
    if 'dx' in dir(first_ds):
        dV = first_ds.domain_width[0]/dims[0] * first_ds.domain_width[1]/dims[1] * first_ds.domain_width[2]/dims[2]
    else:
        dV = np.prod(first_ds.domain_width / first_ds.domain_dimensions)
    
    return mean_fields, dV, mean_fields["density"].shape

# =============================================================================
# CONSTRUCCIÓN DE W (OPTIMIZADA)
# =============================================================================
def build_compressible_W(mean_fields, cell_volume, dims):
    nx, ny = dims
    n_points = nx * ny
    
    rho_mean = mean_fields["density"].flatten()
    T_mean = mean_fields["Temp"].flatten()
    
    # Cálculo vectorizado de pesos
    diagonal = np.empty(4 * n_points)
    diagonal[0::4] = T_mean / (GAMMA * rho_mean * MACH**2)    # ρ
    diagonal[1::4] = rho_mean                                  # vx
    diagonal[2::4] = rho_mean                                  # vy
    diagonal[3::4] = rho_mean / (GAMMA * (GAMMA-1) * T_mean * MACH**2)  # T
    
    diagonal *= cell_volume  # Aplicar volumen de celda
    
    return diags(diagonal, 0, format='csr')

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    # 1. Cargar todos los snapshots
    root = "../../../Simulations/S0.1"
    files = sorted([f for f in os.listdir(root) if "pltAMR2NSW_" in f], 
                  key=lambda x: int(x.split("_")[-1]))
    print(f"\nTotal de snapshots a procesar: {len(files)}")
    
    # 2. Cálculo de promedios con ROI (en batches)
    mean_fields, dV, dims = calculate_mean_fields_batched(files, root)
    
    # 3. Construcción de W
    print("\nConstruyendo matriz W...")
    W = build_compressible_W(mean_fields, dV, dims)
    
    # 4. Guardar resultados
    output_dir = "../W"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"W_compressible_gamma{GAMMA}_Mach{MACH}_FULL.npz")
    
    from scipy.sparse import save_npz
    save_npz(output_path, W)
    
    print(f"\n✅ Matriz W guardada en: {output_path}")
    print(f"Dimensiones: {W.shape}")
    print(f"Elementos no cero: {W.count_nonzero():,}")
    print(f"Memoria usada: {W.data.nbytes / (1024**2):.2f} MB")