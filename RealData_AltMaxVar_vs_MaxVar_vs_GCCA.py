import numpy as np
import time
import tracemalloc
import os
import sys
import json
import tensorflow as tf
import GPUtil
from datetime import datetime
from scipy.sparse import load_npz, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ====================================================
# Create experiment folder with timestamp
# ====================================================
exp_dir = f"exp/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
os.makedirs(exp_dir, exist_ok=True)

# ====================================================
# Helper function: Measure memory usage
# ====================================================
def measure_memory_usage():
    peak_cpu_memory = tracemalloc.get_traced_memory()[1] / (1024 * 1024)  # MB
    peak_gpu_memory = max([gpu.memoryUsed for gpu in GPUtil.getGPUs()], default=0)  # MB
    return peak_cpu_memory + peak_gpu_memory

# ====================================================
# Load and process PMI matrices with Truncated SVD
# ====================================================
def load_and_preprocess_pmi(file_path, target_dim=300):
    print(f"Loading {file_path}...")
    X_sparse = load_npz(file_path)
    print(f"Original Shape: {X_sparse.shape}")

    svd = TruncatedSVD(n_components=target_dim, random_state=42)
    X_reduced = svd.fit_transform(X_sparse)

    print(f"Reduced Shape: {X_reduced.shape}\n")
    return tf.convert_to_tensor(X_reduced, dtype=tf.float32)

pmi_en = load_and_preprocess_pmi("pmi_en.npz", target_dim=300)
pmi_fr = load_and_preprocess_pmi("pmi_fr.npz", target_dim=300)
pmi_es = load_and_preprocess_pmi("pmi_es.npz", target_dim=300)
datasets = [pmi_en, pmi_fr, pmi_es]

# ====================================================
# GCCA Algorithm with TensorFlow GPU acceleration
# ====================================================
def gcca_gpu(datasets, k, max_iter=1000, tol=1e-4):
    I = len(datasets)
    N = datasets[0].shape[0]

    A_matrices = [tf.Variable(tf.random.normal([X.shape[1], k], dtype=tf.float32)) for X in datasets]
    G = tf.Variable(tf.random.normal([N, k], dtype=tf.float32))

    tracemalloc.start()
    start_time = time.time()

    for iteration in range(max_iter):
        G_old = tf.identity(G)

        # Update A_matrices
        for i in range(I):
            X = datasets[i]
            A_matrices[i].assign(tf.linalg.lstsq(X, G, l2_regularizer=1e-6))

        # Update G
        G.assign(tf.reduce_mean([tf.matmul(X, A) for X, A in zip(datasets, A_matrices)], axis=0))

        # Check convergence
        delta_G = tf.norm(G - G_old).numpy()
        progress = (iteration + 1) / max_iter * 100
        sys.stdout.write(f"\rGCCA [{iteration+1}/{max_iter}] {progress:.2f}% | ΔG: {delta_G:.6e}")
        sys.stdout.flush()

        if delta_G < tol:
            break

    total_time = time.time() - start_time
    total_memory = measure_memory_usage()
    tracemalloc.stop()

    return G.numpy(), total_time, total_memory

# ====================================================
# MaxVar GCCA Algorithm
# ====================================================
def maxvar_gcca_gpu(datasets, k):
    N = datasets[0].shape[0]
    M = tf.zeros([N, N], dtype=tf.float32)

    tracemalloc.start()
    start_time = time.time()

    for X in datasets:
        X_pinv = tf.linalg.pinv(X)
        M += tf.matmul(X, X_pinv)

    S, U, V = tf.linalg.svd(M)
    G_opt = U[:, :k]

    total_time = time.time() - start_time
    total_memory = measure_memory_usage()
    tracemalloc.stop()

    return G_opt.numpy(), total_time, total_memory

# ====================================================
# Alternating MaxVar GCCA Algorithm
# ====================================================
def altmaxvar_gcca_gpu(datasets, k, max_iter=50, tol=1e-6):
    I = len(datasets)
    N = datasets[0].shape[0]

    A_matrices = [tf.Variable(tf.random.normal([X.shape[1], k], dtype=tf.float32)) for X in datasets]
    G = tf.Variable(tf.random.normal([N, k], dtype=tf.float32))

    tracemalloc.start()
    start_time = time.time()

    for r in range(max_iter):
        G_old = tf.identity(G)

        for i in range(I):
            X = datasets[i]
            A_matrices[i].assign(tf.linalg.lstsq(X, G, l2_regularizer=1e-6))

        G.assign(tf.reduce_mean([tf.matmul(X, A) for X, A in zip(datasets, A_matrices)], axis=0))

        # Convergence check
        delta_G = tf.norm(G - G_old).numpy()
        progress = (r + 1) / max_iter * 100
        sys.stdout.write(f"\rAltMaxVar [{r+1}/{max_iter}] {progress:.2f}% | ΔG: {delta_G:.6e}")
        sys.stdout.flush()

        if delta_G < tol:
            break

    total_time = time.time() - start_time
    total_memory = measure_memory_usage()
    tracemalloc.stop()

    return G.numpy(), total_time, total_memory

# ====================================================
# Run Experiments and Save Results
# ====================================================
methods = {
    "GCCA": gcca_gpu,
    "MaxVar GCCA": maxvar_gcca_gpu,
    "AltMaxVar GCCA": altmaxvar_gcca_gpu
}
results = {}

print("\nEvaluating GCCA, MaxVar GCCA, and AltMaxVar GCCA...\n")
for method_name, method_func in methods.items():
    projections, exec_time, mem_usage = method_func(datasets, k=300)
    
    results[method_name] = {"time": exec_time, "memory": mem_usage, "projection": projections}
    print(f"{method_name}: Time={exec_time:.4f}s, Memory={mem_usage:.2f}MB")

# Save results
np.save(f"{exp_dir}/results_gcca.npy", results)

# ====================================================
# Compute Cosine Similarity for Evaluation
# ====================================================
for method, data in results.items():
    cos_sim = cosine_similarity(data["projection"])
    avg_sim = np.mean(cos_sim)
    print(f"{method}: Avg Cosine Similarity={avg_sim:.4f}")

# ====================================================
# Save Cosine Similarity Results
# ====================================================
with open(f"{exp_dir}/cosine_similarity_results.json", "w") as f:
    json.dump({method: np.mean(cosine_similarity(data["projection"])).tolist() for method, data in results.items()}, f)

print("\nGCCA Evaluation Completed!")