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
BUFFER_SIZE = 1  # Procesar frecuencias en bloques (ajustar según RAM)

def load_frequency_data(freq_idx, Q_hat_files, spatial_dims):
    """Carga Q̂ para una frecuencia específica de todos los bloques"""
    nx, ny = spatial_dims
    n_vars = len(VARIABLES)
    M = nx * ny * n_vars
    N = len(Q_hat_files)  # Número de bloques
    
    Q_k = np.zeros((M, N), dtype=np.complex128)
    
    for block_idx, block_files in enumerate(Q_hat_files):
        # Cargar todas las variables para este bloque y frecuencia
        var_data = []
        for var_file in block_files:
            data = np.load(var_file)[:, :, freq_idx]  # Forma (nx, ny)
            var_data.append(data.flatten())  # Aplanar a (nx*ny,)
            data = None
        
        # Concatenar variables verticalmente: [ρ; vx; vy; T]
        Q_k[:, block_idx] = np.concatenate(var_data)
    
    return Q_k

def process_frequency(args):
    freq_idx, Q_hat_files, W_sparse, spatial_dims = args
    
    try:
        # 1. Cargar Q̂ (shape: M×N)
        Q_k = load_frequency_data(freq_idx, Q_hat_files, spatial_dims)  # (M, N)
        N = Q_k.shape[1]

        # 2. Calcular W Q̂ (eficiente para W sparse)
        if isinstance(W_sparse, np.ndarray):
            # W es diagonal (vector)
            W_Q = Q_k * W_sparse.reshape(-1, 1)  # Broadcasting
        else:
            # W es matriz sparse (ej: CSR)
            W_Q = W_sparse @ Q_k  # Multiplicación sparse-densa

        # 3. Calcular Q̂^H W Q̂ (N×N)
        weighted_C = Q_k.conj().T @ W_Q  # (N, N)

        # 4. Resolver problema de autovalores
        eigvals, eigvecs = eigh(weighted_C)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # 5. Modos SPOD: Φ = Q̂ Ψ̂ Λ^{-1/2}
        Phi = Q_k @ eigvecs @ np.diag(1.0 / np.sqrt(eigvals))

        return freq_idx, eigvals, Phi

    except Exception as e:
        print(f"Error en frecuencia {freq_idx}: {str(e)}")
        return freq_idx, None, None
        
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Detectar parámetros
    sample_file = os.path.join(BASE_PATH, VARIABLES[0], "Q_hat_block01.npy")
    sample = np.load(sample_file)
    nx, ny = sample.shape[:2]
    N_FFT = sample.shape[2]
    N_BLOCKS = len([f for f in os.listdir(os.path.join(BASE_PATH, VARIABLES[0])) 
                   if f.startswith("Q_hat_block") and f.endswith(".npy")])
    n_vars = len(VARIABLES)
    M = nx * ny * n_vars
    N = N_BLOCKS

    print(f"Configuración detectada: M = {M}, N = {N}, N_FFT = {N_FFT}")

    # 2. Preparar estructura de archivos
    Q_hat_files = [
        [os.path.join(BASE_PATH, var, f"Q_hat_block{block_idx:02d}.npy") 
         for var in VARIABLES]
        for block_idx in range(1, N_BLOCKS + 1)
    ]

    # 3. Cargar W como matriz diagonal (sparse)
    W_diag = load_npz(W_PATH).diagonal()  # Vector (M,)

    # 4. Procesar en lotes
    eigenvalues = np.zeros((N_FFT, N))
    
    with h5py.File(os.path.join(OUTPUT_DIR, "spod_modes.h5"), 'w') as hf:
        modes_dset = hf.create_dataset(
            "modes", 
            shape=(N_FFT, M, N), 
            dtype=np.complex128,
            chunks=(1, M, N),
            compression='gzip'
        )

        for freq_batch in [range(i, min(i + BUFFER_SIZE, N_FFT)) 
                         for i in range(0, N_FFT, BUFFER_SIZE)]:
            
            print(f"\nProcesando frecuencias {freq_batch.start} a {freq_batch.stop - 1}...")
            
            with Pool(min(cpu_count(), len(freq_batch))) as pool:
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
    print("\n✅ ¡Proceso completado!")

if __name__ == "__main__":
    main()
