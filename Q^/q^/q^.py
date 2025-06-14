# -*- coding: utf-8 -*-
"""
Script para procesamiento por bloques usando el método de Welch
"""

import yt
import numpy as np
import os
from scipy.ndimage import uniform_filter1d
from multiprocessing import Pool
from tqdm import tqdm
import gc
from scipy.signal import get_window

# Configuración global
MAX_WORKERS = 48
CHUNK_SIZE = 4
NUM_BLOCKS = 3  # Número configurable de bloques
BLOCK_OVERLAP = 0.5  # 50% de solapamiento entre bloques

def set_yt_threads(num_threads):
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)

def process_block(args):
    """Procesa un bloque completo con seguimiento dinámico de llama"""
    block_idx, snapshot_files, root, field = args
    try:
        set_yt_threads(1)
        
        # Configuración de bloques (igual que antes)
        total_snapshots = len(snapshot_files)
        block_size = int(total_snapshots / (NUM_BLOCKS - (NUM_BLOCKS-1)*BLOCK_OVERLAP))
        overlap_size = int(block_size * BLOCK_OVERLAP)
        start_idx = int(block_idx * (block_size - overlap_size))
        end_idx = min(start_idx + block_size, total_snapshots)
        block_files = snapshot_files[start_idx:end_idx]

        # Cargar primer snapshot solo para obtener dimensiones base
        first_ds = yt.load(os.path.join(root, block_files[0]))
        max_level = 2
        ref = int(np.prod(first_ds.ref_factors[0:max_level]))
        low = first_ds.domain_left_edge
        dims = first_ds.domain_dimensions * ref
        nx = dims[0]
        
        # Parámetros para detección de región (ajustar según necesidades)
        threshold = 100.0
        window_size = 5
        preset = 75
        offset = 1125
        
        # Determinar tamaño de ROI dinámica (usando el primer snapshot como referencia)
        temp_cube = first_ds.covering_grid(max_level, left_edge=low, dims=dims, 
                                         fields=[("boxlib", "Temp")])
        temp_profile = temp_cube[("boxlib", "Temp")].d[0, :, 0]
        diff_avg = np.convolve(np.diff(temp_profile), 
                             np.ones(2*window_size+1)/(2*window_size+1), 
                             mode='same')
        base_index = np.where(diff_avg > threshold)[0][0]
        roi_height = offset + preset
        ny = roi_height  # Altura fija de la ROI
        
        # Inicializar matriz para el bloque con ROI de tamaño constante
        block_data = np.zeros((nx, ny, len(block_files)))
        
        # Procesar cada snapshot con detección dinámica
        for t, file in enumerate(tqdm(block_files, desc=f"Bloque {block_idx+1}")):
            ds = yt.load(os.path.join(root, file))
            cube = ds.covering_grid(max_level, left_edge=low, dims=dims, 
                                  fields=[("boxlib", field), ("boxlib", "Temp")])
            
            # Detección dinámica de la base de la llama
            temp_matrix = cube[("boxlib", "Temp")].d[:, :, 0]
            profile = temp_matrix[0, :]
            diff_avg = np.convolve(np.diff(profile), 
                                 np.ones(2*window_size+1)/(2*window_size+1), 
                                 mode='same')
            base_candidates = np.where(diff_avg > threshold)[0]
            
            if len(base_candidates) == 0:
                print(f"Advertencia: No se detectó base de llama en {file}")
                base_index = ny // 2  # Valor por defecto
            else:
                base_index = base_candidates[0]
            
            lower_bound = max(0, base_index - preset)
            upper_bound = min(temp_matrix.shape[1], base_index + offset)
            
            # Extraer región de interés con tamaño consistente
            roi = cube[("boxlib", field)].d[:, lower_bound:upper_bound, 0]
            
            # Aplicar padding si es necesario para mantener dimensiones
            if roi.shape[1] < ny:
                pad_width = ((0, 0), (0, ny - roi.shape[1]))
                roi = np.pad(roi, pad_width, mode='constant', constant_values=0)
            
            block_data[:, :, t] = roi[:, :ny]  # Asegurar tamaño exacto
            
            del ds, cube
            gc.collect()
        
        # Procesamiento espectral del bloque (igual que antes)
        window = get_window('hamming', len(block_files))
        n_freq = len(block_files) // 2 + 1
        Q_hat_block = np.zeros((nx, ny, n_freq), dtype=np.complex128)
        
        for i in range(nx):
            for j in range(ny):
                signal = block_data[i, j, :]
                signal_avg = uniform_filter1d(signal, size=5, mode="reflect")
                Q = (signal - signal_avg) * window
                Q_hat_block[i, j, :] = np.fft.rfft(Q)
        
        # Guardar resultados
        os.makedirs(f"q^/{field}", exist_ok=True)
        output_path = f"q^/{field}/q^_block{block_idx+1:02d}.npy"
        np.save(output_path, Q_hat_block)
        
        return output_path
        
    except Exception as e:
        print(f"Error procesando bloque {block_idx} ({field}): {str(e)}")
        return None

def process_variable(field, files, root):
    """Procesa todos los bloques para una variable"""
    print(f"\n{'='*40}\nProcesando variable: {field}\n{'='*40}")
    
    # Procesar bloques en paralelo
    with Pool(min(MAX_WORKERS, NUM_BLOCKS)) as pool:
        args_list = [(i, files, root, field) for i in range(NUM_BLOCKS)]
        results = list(tqdm(pool.imap(process_block, args_list, chunksize=CHUNK_SIZE),
                      total=NUM_BLOCKS, desc=f"Procesando {field}"))
    
    # Verificar resultados
    valid_results = [r for r in results if r is not None]
    print(f"\n{len(valid_results)}/{NUM_BLOCKS} bloques procesados exitosamente para {field}")

def main():
    set_yt_threads(1)
    root = "../../Simulations/S0.1"
    fields = ["y_velocity", "Temp"]
    
    files = [f for f in os.listdir(root) if "pltAMR2NSW_" in f]
    files = sorted(files, key=lambda s: int(s.split("_")[-1]))
    
    for field in fields:
        process_variable(field, files, root)

if __name__ == "__main__":
    main()
