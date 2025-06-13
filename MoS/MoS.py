import numpy as np
from scipy.sparse import load_npz
from scipy.linalg import eigh
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import gc
import h5py

# Configuration
BASE_PATH = "../Q_hat/Q_hat"
W_PATH = "../W/W_matrices/W_compressible_gamma1.354_Mach1.7_FULL.npz"
VARIABLES = ["density", "x_velocity", "y_velocity", "Temp"]
OUTPUT_DIR = "../SPOD_results"
MAX_RAM_USAGE = 20  # GB
BUFFER_SIZE = 1     # Adjust based on available RAM

def load_frequency_data(freq_idx, Q_hat_files, spatial_dims):
    """Load Q̂ for a specific frequency from all blocks"""
    nx, ny = spatial_dims
    n_vars = len(VARIABLES)
    M = nx * ny * n_vars
    N = len(Q_hat_files)  # Number of blocks
    
    Q_k = np.zeros((M, N), dtype=np.complex128)
    
    for block_idx, block_files in enumerate(Q_hat_files):
        # Load all variables for this block and frequency
        var_data = []
        for var_file in block_files:
            data = np.load(var_file)[:, :, freq_idx]  # Shape (512, 1200)
            var_data.append(data.flatten())  # Flatten to (614400,)
            data = None  # Free memory
        
        # Concatenate variables vertically: [ρ; vx; vy; T]
        Q_k[:, block_idx] = np.concatenate(var_data)  # Shape (2457600,)
    
    return Q_k

def process_frequency(args):
    freq_idx, Q_hat_files, W_diag, spatial_dims = args
    
    try:
        # 1. Load Q̂ for this frequency
        Q_k = load_frequency_data(freq_idx, Q_hat_files, spatial_dims)
        N = Q_k.shape[1]  # Number of blocks
        
        # 2. Compute covariance matrix (without W)
        C_snap = (1/(N-1)) * (Q_k @ Q_k.conj().T)  # (M, M)
        
        # 3. Solve weighted eigenvalue problem: Q̂ᴴ W Q̂ Ψ̂ = Ψ̂ Λ
        # Compute W_Q = W Q̂
        W_Q = Q_k * W_diag[None, :]  # (M, N)
        
        # Compute Q̂ᴴ W Q̂ = Q̂ᴴ (W Q̂)
        weighted_C = Q_k.conj().T @ W_Q  # (N, N)
        
        # Eigenvalue decomposition
        eigvals, eigvecs = eigh(weighted_C)
        idx = np.argsort(eigvals)[::-1]  # Sort descending
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # 4. Compute SPOD modes Φ = Q̂ Ψ̂ Λ^{-1/2}
        Phi = Q_k @ eigvecs @ np.diag(1/np.sqrt(eigvals))
        
        return freq_idx, eigvals, Phi
        
    except Exception as e:
        print(f"Error processing frequency {freq_idx}: {str(e)}")
        return freq_idx, None, None
    finally:
        gc.collect()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Detect parameters
    sample = np.load(os.path.join(BASE_PATH, VARIABLES[0], "Q_hat_block01.npy"))
    N_BLOCKS = len([f for f in os.listdir(os.path.join(BASE_PATH, VARIABLES[0])) 
                 if f.startswith("Q_hat_block") and f.endswith(".npy")])
    nx, ny = sample.shape[:2]
    N_FFT = sample.shape[2]
    n_vars = len(VARIABLES)
    M = nx * ny * n_vars
    N = N_BLOCKS
    
    print(f"Detected configuration: {M}×{N} (Frequencies: {N_FFT})")
    
    # 2. Prepare file structure
    Q_hat_files = []
    for block_idx in range(1, N_BLOCKS+1):
        block_files = [os.path.join(BASE_PATH, var, f"Q_hat_block{block_idx:02d}.npy") for var in VARIABLES]
        Q_hat_files.append(block_files)
    
    # 3. Load only diagonal of W
    W_diag = load_npz(W_PATH).diagonal()
    
    # 4. Batch processing
    eigenvalues = np.zeros((N_FFT, N))
    
    with h5py.File(os.path.join(OUTPUT_DIR, "spod_modes.h5"), 'w') as hf:
        modes_dset = hf.create_dataset(
            "modes", 
            shape=(N_FFT, M, N), 
            dtype=np.complex128,
            chunks=(1, M, N),
            compression='gzip'
        )
        
        # Process in batches for memory control
        for freq_batch in [range(i, min(i+BUFFER_SIZE, N_FFT)) 
                          for i in range(0, N_FFT, BUFFER_SIZE)]:
            
            print(f"\nProcessing frequencies {freq_batch.start} to {freq_batch.stop-1}...")
            
            # Parallel processing within batch
            with Pool(min(64, cpu_count(), len(freq_batch))) as pool:
                args = [(f, Q_hat_files, W_diag, (nx, ny)) for f in freq_batch]
                
                for result in tqdm(pool.imap_unordered(process_frequency, args),
                                total=len(freq_batch),
                                desc="Progress"):
                    freq_idx, eigvals, Phi = result
                    
                    if eigvals is not None:
                        eigenvalues[freq_idx] = eigvals
                        modes_dset[freq_idx] = Phi
    
    # Save eigenvalues
    np.save(os.path.join(OUTPUT_DIR, "spod_eigenvalues.npy"), eigenvalues)
    
    print("\n✅ Process completed successfully!")

if __name__ == "__main__":
    main()
