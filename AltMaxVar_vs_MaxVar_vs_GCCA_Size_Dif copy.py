from util import generate_sparse_nonnegative_data
from CoreAlgorithm import gcca, maxvar_gcca, altmaxvar_gcca
import numpy as np
import time
import tracemalloc
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Create experiment directory
exp_root = os.getcwd()
exp_root = os.path.join(exp_root, "exp")
if not os.path.exists(exp_root):  
    os.makedirs(exp_root)  # Adapt to Mac OS, ensure directory exists

# Generate current timestamp experiment folder
exp_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_dir = os.path.join(exp_root, exp_time)
os.makedirs(exp_dir, exist_ok=True)  # Prevent creation failure

# Increase large-scale dataset
N_list = [20, 100, 500, 2000, 10000, 50000, 100000]  # Sample size
d_list = [120, 600, 1200, 2400, 5000, 10000, 20000]  # Feature number
k_list = [50, 200, 500, 1000, 2000, 5000, 10000]  # Shared dimension

noise_std = 1
outliers_noise_scale = -1
sparsity_std = 0.01  # Increase sparsity
nonneg = True  # Only test non-negative data
result_in_dense = True  # Ensure data format is numpy array

# Only test 'nonneg' and 'sparse'
constraints = ['nonneg', 'sparse', 'both']

# Store experiment results
results = {"GCCA": {"time": [], "memory": []},
           "MaxVar GCCA": {"time": [], "memory": []},
           "AltMaxVar GCCA": {"time": [], "memory": []}}

print("\n================= GCCA vs MaxVar vs AltMaxVar GCCA =================\n")

for i, N in enumerate(N_list):
    print(f"\n[ Running for Sample Size N={N}, Feature Dim d={d_list[i]}, Shared k={k_list[i]} ]")
    
    # Generate data
    datasets = generate_sparse_nonnegative_data(
        N=N, I=3, d_list=[d_list[i], d_list[i], d_list[i]], k=k_list[i],
        noise_std=noise_std, sparsity_std=sparsity_std, 
        nonneg=True, outliers_noise_scale=outliers_noise_scale, result_in_dense=True
    )
    
    np.save(os.path.join(exp_dir, f"datasets_N{N}.npy"), datasets)

    # ===== GCCA =====
    projections_gcca, A_matrices_gcca, G_gcca, time_gcca, mem_gcca = gcca(datasets, k=k_list[i])
    results["GCCA"]["time"].append(time_gcca)
    results["GCCA"]["memory"].append(mem_gcca)
    np.save(os.path.join(exp_dir, f"GCCA_G_N{N}.npy"), G_gcca)

    # ===== MaxVar GCCA =====
    projections_maxvar, A_matrices_maxvar, G_maxvar, time_maxvar, mem_maxvar = maxvar_gcca(datasets, k=k_list[i])
    results["MaxVar GCCA"]["time"].append(time_maxvar)
    results["MaxVar GCCA"]["memory"].append(mem_maxvar)
    np.save(os.path.join(exp_dir, f"MaxVar_G_N{N}.npy"), G_maxvar)

    # ===== AltMaxVar GCCA ('nonneg' & 'sparse') =====
    for constraint in constraints:
        projections_altmaxvar, A_matrices_altmaxvar, G_altmaxvar, time_altmaxvar, mem_altmaxvar = altmaxvar_gcca(
            datasets, k=k_list[i], max_iter=50, inner_iter=10, alpha=0.01, gamma=0.5, constraint=constraint, param=0.1
        )
        results["AltMaxVar GCCA"]["time"].append(time_altmaxvar)
        results["AltMaxVar GCCA"]["memory"].append(mem_altmaxvar)
        np.save(os.path.join(exp_dir, f"AltMaxVar_{constraint}_G_N{N}.npy"), G_altmaxvar)

    # ===== Clear comparison output =====
    print("\n================= Performance Comparison =================")

    # Table header
    header = "| {:^22} | {:^22} | {:^22} |".format("Method", "Execution Time (sec)", "Memory Usage (MB)")
    separator = "-" * len(header)

    print(f"| Sample Size (N={N}) | Feature Dim: {d_list[i]} | Shared Dim: {k_list[i]} |")
    print(separator)
    print(header)
    print(separator)

    # GCCA result
    print("| {:<22} | {:>20.4f} | {:>20.2f} |".format("GCCA", time_gcca, mem_gcca))

    # MaxVar GCCA result
    print("| {:<22} | {:>20.4f} | {:>20.2f} |".format("MaxVar GCCA", time_maxvar, mem_maxvar))

    # AltMaxVar GCCA result
    for j, constraint in enumerate(constraints):
        print("| {:<22} | {:>20.4f} | {:>20.2f} |".format(
            f"AltMaxVar ({constraint})", 
            results['AltMaxVar GCCA']['time'][-len(constraints) + j], 
            results['AltMaxVar GCCA']['memory'][-len(constraints) + j]
        ))

    print(separator)

# ====== Execution efficiency (time & memory vs. data scale) chart ======
plt.figure(figsize=(8, 5))
for method in results:
    plt.plot(N_list, results[method]["time"], label=f"{method} - Time", marker="o", linestyle="solid")
plt.xlabel("Sample Size (N)")
plt.ylabel("Execution Time (sec)")
plt.title("Execution Time vs. Sample Size")
plt.legend()
plt.grid()
plt.savefig(os.path.join(exp_dir, "Execution_Time_vs_Sample_Size.png"))
plt.show(block=False)
plt.pause(0.1)

plt.figure(figsize=(8, 5))
for method in results:
    plt.plot(N_list, results[method]["memory"], label=f"{method} - Memory", marker="s", linestyle="dashed")
plt.xlabel("Sample Size (N)")
plt.ylabel("Memory Usage (MB)")
plt.title("Memory Usage vs. Sample Size")
plt.legend()
plt.grid()
plt.savefig(os.path.join(exp_dir, "Memory_Usage_vs_Sample_Size.png"))
plt.show(block=False)
plt.pause(0.1)

print(f"\nAll experiment results saved in: {exp_dir}\n")

