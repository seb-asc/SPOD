# prepare_Q_hat_matrices.py
import numpy as np
import os
from tqdm import tqdm
import h5py
import concurrent.futures
from math import ceil

BASE_PATH = "../Q^/q^"
VARIABLES = ["density", "x_velocity", "y_velocity", "Temp"]
OUTPUT_PATH = "../Q^/Q^.h5"

def get_system_dimensions():
    """Obtiene las dimensiones del sistema a partir de los archivos de datos"""
    sample_path = os.path.join(BASE_PATH, VARIABLES[0], "q^_block01.npy")
    sample = np.load(sample_path)
    nx, ny, n_freq = sample.shape
    n_blocks = len([f for f in os.listdir(os.path.join(BASE_PATH, VARIABLES[0])) 
                   if f.startswith("q^_block")])
    return nx, ny, n_freq, n_blocks

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    print("Obteniendo dimensiones del sistema...")
    nx, ny, n_freq, n_blocks = get_system_dimensions()
    M = nx * ny * len(VARIABLES)
    
    print(f"Dimensiones: nx={nx}, ny={ny}, frecuencias={n_freq}, bloques={n_blocks}")
    print(f"Tamaño de Q^: {M} x {n_blocks} (complejo 128 bits)")
    
    # Calcular tamaño de memoria por frecuencia
    size_per_freq_bytes = M * n_blocks * 16  # complex128 = 16 bytes
    available_mem_bytes = 10 * 1024**3  # 18 GB disponibles
    chunk_size = max(1, min(200, available_mem_bytes // size_per_freq_bytes))
    print(f"\nTamaño por frecuencia: {size_per_freq_bytes/1024**2:.2f} MB")
    print(f"Chunk size calculado: {chunk_size} frecuencias")
    print(f"Memoria máxima por chunk: {chunk_size * size_per_freq_bytes/1024**3:.2f} GB")

    # Pre-cargar arrays con memory mapping
    print("\nCargando archivos con memory mapping...")
    mmap_arrays = {}
    for var in tqdm(VARIABLES, desc="Variables"):
        mmap_arrays[var] = {}
        for block_idx in tqdm(range(1, n_blocks + 1), desc="Bloques", leave=False):
            file_path = os.path.join(BASE_PATH, var, f"q^_block{block_idx:02d}.npy")
            mmap_arrays[var][block_idx] = np.load(file_path, mmap_mode='r')

    # Función para procesar una frecuencia
    def process_frequency(freq_idx):
        Q_hat = np.zeros((M, n_blocks), dtype=np.complex128)
        for block_idx in range(1, n_blocks + 1):
            var_data = []
            for var in VARIABLES:
                data = mmap_arrays[var][block_idx][:, :, freq_idx].flatten()
                var_data.append(data)
            Q_hat[:, block_idx - 1] = np.concatenate(var_data)
        return Q_hat

    # Configuración de paralelismo
    n_workers = min(48, os.cpu_count())
    print(f"\nUsando {n_workers} workers para procesamiento paralelo")
    
    with h5py.File(OUTPUT_PATH, 'w') as hf:
        dset = hf.create_dataset(
            "Q_hat_matrices",
            shape=(n_freq, M, n_blocks),
            dtype=np.complex128,
            chunks=(1, M, n_blocks),
            compression='gzip'
        )
        
        print("\nProcesando frecuencias en chunks...")
        with tqdm(total=n_freq, desc="Progreso") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                for start_idx in range(0, n_freq, chunk_size):
                    end_idx = min(start_idx + chunk_size, n_freq)
                    chunk_range = range(start_idx, end_idx)
                    
                    # Procesar chunk actual
                    futures = {executor.submit(process_frequency, i): i for i in chunk_range}
                    chunk_results = {}
                    
                    for future in concurrent.futures.as_completed(futures):
                        freq_idx = futures[future]
                        try:
                            chunk_results[freq_idx] = future.result()
                        except Exception as e:
                            print(f"\nError procesando frecuencia {freq_idx}: {e}")
                        finally:
                            pbar.update(1)
                    
                    # Escribir resultados en orden
                    for freq_idx in chunk_range:
                        dset[freq_idx] = chunk_results[freq_idx]
                    
                    # Liberar memoria explícitamente
                    del futures, chunk_results

    print(f"\n✅ Matrices Q^ guardadas en {OUTPUT_PATH}")

if __name__ == "__main__":
    main()