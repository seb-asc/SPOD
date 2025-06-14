# save_W_diagonal.py
import numpy as np
from scipy.sparse import load_npz
import os

W_PATH = "../W/W_matrix.npz"
OUTPUT_PATH = "../W/W_diagonal.npy"

def extract_diagonal():
    """Extrae la diagonal de W y la guarda como vector columna"""
    print("Cargando matriz W...")
    W_sparse = load_npz(W_PATH)
    
    print("Extrayendo diagonal...")
    W_diag = W_sparse.diagonal().astype(np.float64, copy=False)
    del W_sparse
    
    # Guardar como vector columna (M,1)
    W_diag = W_diag.reshape(-1, 1)
    np.save(OUTPUT_PATH, W_diag)
    print(f"Diagonal guardada en {OUTPUT_PATH} (shape: {W_diag.shape})")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    extract_diagonal()