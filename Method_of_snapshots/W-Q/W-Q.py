# compute_WQ_optimized.py
import numpy as np
import h5py
import os
from tqdm import tqdm

W_PATH = "../../W/W_diagonal.npy"
Q_HAT_PATH = "../../Q^/Q^.h5"
OUTPUT_DIR = "../Method_of_snapshots/W-Q"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Cargar diagonal W
    W_diag = np.load(W_PATH).flatten()  # Convertir a vector (M,)
    
    # Procesar cada frecuencia
    with h5py.File(Q_HAT_PATH, 'r') as hf:
        n_freq = hf['Q_hat_matrices'].shape[0]
        
        for freq_idx in tqdm(range(n_freq), desc="Calculando W_Q"):
            Q_k = hf['Q_hat_matrices'][freq_idx]
            W_Q = Q_k * W_diag.reshape(-1, 1)  # Broadcasting
            np.save(os.path.join(OUTPUT_DIR, f"WQ_freq{freq_idx:04d}.npy"), W_Q)

if __name__ == "__main__":
    main()
