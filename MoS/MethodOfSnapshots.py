import numpy as np
from scipy.sparse import load_npz
from scipy.linalg import eigh
import os
from tqdm import tqdm

# Configuración
BASE_PATH = "Q_hat"
W_PATH = "W_compressible_gamma1.354_Mach1.7_ROI.npz"
VARIABLES = ["density", "x_velocity", "y_velocity", "Temp"]
N_BLOCKS = 3
N_FFT = 331  # Número de frecuencias (N/2 + 1)
SPATIAL_DIMS = (1200, 512)  # nx, ny

def load_all_Q_hat_frequencies():
    """Carga todos los Q_hat_block y los organiza por frecuencia"""
    # Inicializar estructura: frec × bloques × variables × espacio
    Q_hat_freq = np.zeros((N_FFT, N_BLOCKS, len(VARIABLES), dtype=object)
    
    for block_idx in range(1, N_BLOCKS + 1):
        for var_idx, variable in enumerate(VARIABLES):
            # Cargar bloque para esta variable
            block_path = os.path.join(BASE_PATH, variable, f"Q_hat_block{block_idx:02d}.npy")
            block_data = np.load(block_path)  # Forma: (nx, ny, N_FFT)
            
            # Reorganizar por frecuencia
            for freq_idx in range(N_FFT):
                Q_hat_freq[freq_idx, block_idx-1, var_idx] = block_data[:, :, freq_idx]
    
    return Q_hat_freq

def compute_SPOD_modes(Q_hat_freq, W):
    """Calcula modos SPOD para cada frecuencia usando método de snapshots"""
    # Inicializar estructuras para resultados
    eigenvalues = np.zeros((N_FFT, N_BLOCKS))
    modes = np.zeros((N_FFT, N_BLOCKS), dtype=object)
    
    # Cargar matriz de pesos
    W = load_npz(W_PATH)
    W_diag = W.diagonal()  # Extraer diagonal como array 1D
    
    # Procesar cada frecuencia
    for freq_idx in tqdm(range(N_FFT), desc="Procesando frecuencias"):
        # Construir Q_hat para esta frecuencia (concatenar variables y bloques)
        Q_k = []
        for block_idx in range(N_BLOCKS):
            # Concatenar variables para este bloque
            block_vars = []
            for var_idx in range(len(VARIABLES)):
                block_vars.append(Q_hat_freq[freq_idx, block_idx, var_idx].flatten())
            Q_block = np.concatenate(block_vars)
            Q_k.append(Q_block)
        
        # Q_hat(f_k) ∈ C^{M×N} (M=2,457,600, N=3)
        Q_k = np.column_stack(Q_k)  # Forma: (2,457,600 × 3)
        
        # Paso 1: Matriz de covarianza reducida C_snap = (1/(N-1)) Q_k^H W Q_k
        W_Q = Q_k.conj().T * W_diag[:, None]  # Equivalente a diag(W) @ Q_k
        C_snap = (1/(N_BLOCKS-1)) * (Q_k.conj().T @ W_Q)
        
        # Paso 2: Descomposición espectral de C_snap (3×3)
        eigvals, eigvecs = eigh(C_snap)  # eigvecs ya normalizados
        
        # Ordenar en orden descendente
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Paso 3: Calcular modos SPOD Φ = Q_k Ψ Λ^{-1/2}
        Phi = Q_k @ eigvecs @ np.diag(1/np.sqrt(eigvals))
        
        # Almacenar resultados
        eigenvalues[freq_idx] = eigvals
        modes[freq_idx] = Phi
    
    return eigenvalues, modes

def save_SPOD_results(eigenvalues, modes):
    """Guarda los modos y eigenvalues en formato .npy"""
    os.makedirs("SPOD_results", exist_ok=True)
    
    # Guardar eigenvalues (todas las frecuencias)
    np.save("SPOD_results/spod_eigenvalues.npy", eigenvalues)
    
    # Guardar modos por frecuencia
    for freq_idx in range(N_FFT):
        np.save(f"SPOD_results/spod_modes_freq{freq_idx:04d}.npy", modes[freq_idx])

if __name__ == "__main__":
    print("Cargando datos Q_hat...")
    Q_hat_freq = load_all_Q_hat_frequencies()
    
    print("\nCalculando modos SPOD...")
    eigenvalues, modes = compute_SPOD_modes(Q_hat_freq, W_PATH)
    
    print("\nGuardando resultados...")
    save_SPOD_results(eigenvalues, modes)
    
    print("\n¡Proceso completado!")
    print(f"Modos SPOD guardados en directorio 'SPOD_results'")
    print(f"Eigenvalues shape: {eigenvalues.shape}")
    print(f"Ejemplo de eigenvalues para primera frecuencia: {eigenvalues[0]}")
