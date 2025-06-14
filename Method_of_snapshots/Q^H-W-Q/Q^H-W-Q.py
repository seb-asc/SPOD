# compute_final_product_optimized.py
import numpy as np
import h5py
import os
from tqdm import tqdm

Q_HAT_PATH = "../../Q^/Q^.h5"
WQ_PATH = "../W-Q"
OUTPUT_DIR = "../Q^H-W-Q/SPOD_results"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with h5py.File(Q_HAT_PATH, 'r') as Q_hat_file:
        n_freq, M, N = Q_hat_file['Q_hat_matrices'].shape
        
        with h5py.File(os.path.join(OUTPUT_DIR, "spod_results.h5"), 'w') as out_file:
            modes_dset = out_file.create_dataset(
                "modes", 
                shape=(n_freq, M, N), 
                dtype=np.complex128,
                chunks=(1, M, N),
                compression='gzip'
            )
            
            eigenvalues = np.zeros((n_freq, N))
            
            for freq_idx in tqdm(range(n_freq), desc="Procesando SPOD"):
                Q_hat = Q_hat_file['Q_hat_matrices'][freq_idx]
                W_Q = np.load(os.path.join(WQ_PATH, f"WQ_freq{freq_idx:04d}.npy"))
                
                # Cálculo del producto matricial
                weighted_C = Q_hat.conj().T @ W_Q
                
                # Descomposición espectral
                eigvals, eigvecs = np.linalg.eigh(weighted_C)
                idx = np.argsort(eigvals)[::-1]
                eigvals_sorted = eigvals[idx]
                
                # Cálculo de modos SPOD
                scaling = 1.0 / np.sqrt(eigvals_sorted)
                Phi = (Q_hat @ eigvecs[:, idx]) * scaling.reshape(1, -1)
                
                # Almacenamiento
                eigenvalues[freq_idx] = eigvals_sorted
                modes_dset[freq_idx] = Phi
    
    np.save(os.path.join(OUTPUT_DIR, "spod_eigenvalues.npy"), eigenvalues)
    print("\n✅ ¡Proceso SPOD completado!")

if __name__ == "__main__":
    main()
