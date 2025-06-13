import numpy as np
from scipy.sparse import load_npz
from scipy.linalg import eigh
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import gc
import h5py

# Configuración
BASE_PATH = "../Q_hat/Q_hat"
W_PATH = "../W/W_matrices/W_compressible_gamma1.354_Mach1.7_FULL.npz"
VARIABLES = ["density", "x_velocity", "y_velocity", "Temp"]
OUTPUT_DIR = "../SPOD_results"
MAX_RAM_USAGE = 20  # GB
BUFFER_SIZE = 1   # Ajustar según RAM disponible

def load_frequency_data(freq_idx, Q_hat_files, spatial_dims):
    """Carga datos para una frecuencia específica"""
    nx, ny = spatial_dims
    n_vars = len(VARIABLES)
    M = nx * ny * n_vars
    N = len(Q_hat_files)
    
    Q_k = np.zeros((M, N), dtype=np.complex128)
    
    for block_idx, block_files in enumerate(Q_hat_files):
        block_data = []
        for var_file in block_files:
            # Carga directa sin mmap (solución al problema)
            data = np.load(var_file)
            block_data.append(data[:, :, freq_idx].flatten())
            data = None  # Liberar memoria
            
        Q_k[:, block_idx] = np.concatenate(block_data)
    
    return Q_k

def process_frequency(args):
    """Procesa una frecuencia individual"""
    freq_idx, Q_hat_files, W_diag, spatial_dims = args
    
    try:
        # 1. Cargar datos para esta frecuencia
        Q_k = load_frequency_data(freq_idx, Q_hat_files, spatial_dims)
        
        # 2. Calcular matriz de covarianza
        W_Q = Q_k.conj().T * W_diag[:, None]
        C_snap = (1/(Q_k.shape[1]-1)) * (Q_k.conj().T @ W_Q)
        
        # 3. Descomposición espectral
        eigvals, eigvecs = eigh(C_snap)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # 4. Calcular modos SPOD
        Phi = Q_k @ eigvecs @ np.diag(1/np.sqrt(eigvals))
        
        return freq_idx, eigvals, Phi
        
    except Exception as e:
        print(f"Error procesando frecuencia {freq_idx}: {str(e)}")
        return freq_idx, None, None
    finally:
        gc.collect()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Detectar parámetros
    sample = np.load(os.path.join(BASE_PATH, VARIABLES[0], "Q_hat_block01.npy"))
    N_BLOCKS = len([f for f in os.listdir(os.path.join(BASE_PATH, VARIABLES[0])) 
                 if f.startswith("Q_hat_block") and f.endswith(".npy")])
    nx, ny = sample.shape[:2]
    N_FFT = sample.shape[2]
    n_vars = len(VARIABLES)
    M = nx * ny * n_vars
    N = N_BLOCKS
    
    print(f"Configuración detectada: {M}×{N} (Frecuencias: {N_FFT})")
    
    # 2. Preparar estructura de archivos
    Q_hat_files = []
    for block_idx in range(1, N_BLOCKS+1):
        block_files = [os.path.join(BASE_PATH, var, f"Q_hat_block{block_idx:02d}.npy") for var in VARIABLES]
        Q_hat_files.append(block_files)
    
    # 3. Cargar solo diagonal de W
    W_diag = load_npz(W_PATH).diagonal()
    
    # 4. Procesamiento por lotes
    eigenvalues = np.zeros((N_FFT, N))
    
    with h5py.File(os.path.join(OUTPUT_DIR, "spod_modes.h5"), 'w') as hf:
        modes_dset = hf.create_dataset(
            "modes", 
            shape=(N_FFT, M, N), 
            dtype=np.complex128,
            chunks=(1, M, N),
            compression='gzip'
        )
        
        # Procesar en lotes para control de memoria
        for freq_batch in [range(i, min(i+BUFFER_SIZE, N_FFT)) 
                          for i in range(0, N_FFT, BUFFER_SIZE)]:
            
            print(f"\nProcesando frecuencias {freq_batch.start} a {freq_batch.stop-1}...")
            
            # Procesamiento paralelo dentro del lote
            with Pool(min(64, cpu_count(), len(freq_batch))) as pool:
                args = [(f, Q_hat_files, W_diag, (nx, ny)) for f in freq_batch]
                
                for result in tqdm(pool.imap_unordered(process_frequency, args),
                                total=len(freq_batch),
                                desc="Progreso"):
                    freq_idx, eigvals, Phi = result
                    
                    if eigvals is not None:
                        eigenvalues[freq_idx] = eigvals
                        modes_dset[freq_idx] = Phi
    
    # Guardar autovalores
    np.save(os.path.join(OUTPUT_DIR, "spod_eigenvalues.npy"), eigenvalues)
    
    print("\n✅ Proceso completado exitosamente!")

if __name__ == "__main__":
    main()
