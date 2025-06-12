import numpy as np
from scipy.sparse import load_npz
from scipy.linalg import eigh
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import gc

# Configuración de paths
BASE_PATH = "../Q_hat"
W_PATH = "../W/W_matrices/W_compressible_gamma1.354_Mach1.7_ROI.npz"
VARIABLES = ["density", "x_velocity", "y_velocity", "Temp"]
OUTPUT_DIR = "../SPOD_results"

def detect_parameters():
    """Detecta automáticamente los parámetros del dataset"""
    # Detectar número de bloques
    var_dir = os.path.join(BASE_PATH, VARIABLES[0])
    N_BLOCKS = len([f for f in os.listdir(var_dir) if f.startswith("Q_hat_block") and f.endswith(".npy")])
    
    # Detectar dimensiones usando el primer bloque
    sample_block = np.load(os.path.join(var_dir, "Q_hat_block01.npy"))
    SPATIAL_DIMS = sample_block.shape[:2]
    N_FFT = sample_block.shape[2]
    
    return N_BLOCKS, SPATIAL_DIMS, N_FFT

def load_W_matrix():
    """Carga y verifica la matriz W"""
    W = load_npz(W_PATH)
    print(f"Matriz W cargada. Dimensiones: {W.shape}")
    return W

def process_frequency(args):
    """Procesa una sola frecuencia usando el método de snapshots"""
    freq_idx, Q_hat_freq, W_diag = args
    
    try:
        # Construir Q_k para esta frecuencia (M × N)
        Q_k = []
        for block_idx in range(Q_hat_freq.shape[1]):
            # Concatenar variables para este bloque
            block_vars = []
            for var_idx in range(Q_hat_freq.shape[2]):
                block_vars.append(Q_hat_freq[freq_idx, block_idx, var_idx].flatten())
            Q_block = np.concatenate(block_vars)
            Q_k.append(Q_block)
        
        Q_k = np.column_stack(Q_k)  # Forma: (M × N)
        
        # Paso 1: Matriz de covarianza reducida C_snap = (1/(N-1)) Q_k^H W Q_k
        W_Q = Q_k.conj().T * W_diag[:, None]  # Equivalente a diag(W) @ Q_k
        C_snap = (1/(Q_k.shape[1]-1)) * (Q_k.conj().T @ W_Q)
        
        # Paso 2: Descomposición espectral de C_snap (N×N)
        eigvals, eigvecs = eigh(C_snap)
        
        # Ordenar en orden descendente
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Paso 3: Calcular modos SPOD Φ = Q_k Ψ Λ^{-1/2}
        Phi = Q_k @ eigvecs @ np.diag(1/np.sqrt(eigvals))
        
        # Liberar memoria
        del Q_k, W_Q, C_snap
        gc.collect()
        
        return freq_idx, eigvals, Phi
    
    except Exception as e:
        print(f"Error en frecuencia {freq_idx}: {str(e)}")
        return freq_idx, None, None

def compute_SPOD_modes_parallel(Q_hat_freq, W_diag, N_FFT):
    """Calcula modos SPOD en paralelo para todas las frecuencias"""
    # Configurar paralelización
    n_workers = min(64, cpu_count())
    chunk_size = max(1, N_FFT // (n_workers * 10))  # Balancear carga
    
    print(f"\nCalculando modos SPOD usando {n_workers} cores...")
    
    # Preparar argumentos
    args = [(freq_idx, Q_hat_freq, W_diag) for freq_idx in range(N_FFT)]
    
    # Ejecutar en paralelo
    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(process_frequency, args, chunksize=chunk_size),
                      total=N_FFT))
    
    # Organizar resultados
    eigenvalues = np.zeros((N_FFT, Q_hat_freq.shape[1]))
    modes = [None] * N_FFT
    
    for freq_idx, eigvals, Phi in results:
        if eigvals is not None:
            eigenvalues[freq_idx] = eigvals
            modes[freq_idx] = Phi
    
    return eigenvalues, modes

def main():
    # Crear directorio de salida
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Detectar parámetros automáticamente
    N_BLOCKS, SPATIAL_DIMS, N_FFT = detect_parameters()
    print(f"\nParámetros detectados:")
    print(f"- Número de bloques: {N_BLOCKS}")
    print(f"- Dimensiones espaciales: {SPATIAL_DIMS}")
    print(f"- Número de frecuencias: {N_FFT}")
    
    # Cargar y organizar todos los Q_hat por frecuencia
    print("\nCargando y organizando datos Q_hat...")
    Q_hat_freq = np.zeros((N_FFT, N_BLOCKS, len(VARIABLES)), dtype=object)
    
    for block_idx in range(N_BLOCKS):
        for var_idx, variable in enumerate(VARIABLES):
            block_path = os.path.join(BASE_PATH, variable, f"Q_hat_block{block_idx+1:02d}.npy")
            block_data = np.load(block_path)  # Forma: (nx, ny, N_FFT)
            
            for freq_idx in range(N_FFT):
                Q_hat_freq[freq_idx, block_idx, var_idx] = block_data[:, :, freq_idx]
    
    # Cargar matriz W (solo diagonal)
    W = load_W_matrix()
    W_diag = W.diagonal()
    del W  # Liberar memoria
    gc.collect()
    
    # Calcular modos SPOD en paralelo
    eigenvalues, modes = compute_SPOD_modes_parallel(Q_hat_freq, W_diag, N_FFT)
    
    # Guardar resultados
    print("\nGuardando resultados...")
    np.save(os.path.join(OUTPUT_DIR, "spod_eigenvalues.npy"), eigenvalues)
    
    for freq_idx in tqdm(range(N_FFT), desc="Guardando modos"):
        if modes[freq_idx] is not None:
            np.save(os.path.join(OUTPUT_DIR, f"spod_modes_freq{freq_idx:04d}.npy"), modes[freq_idx])
    
    print("\n¡Proceso completado exitosamente!")
    print(f"Resultados guardados en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
