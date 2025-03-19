import numpy as np
from util import *
from CoreAlgorithm import *

# Set parameters
N = 20  # Sample size
d1, d2, d3 = 120, 120, 120  # Feature dimensions for X, Y, and W
k = 50  # Shared latent factor dimension
noise_std = 1
outliers_noise_scale = -1
sparsity_std=1e-3

# Generate synthetic data (X, Y, W)
datasets = generate_synthetic_gcca_data(N=N, I=3, d_list=[d1, d2, d3], k=k, noise_std=noise_std, outliers_noise_scale=outliers_noise_scale, sparsity_std=sparsity_std)
X, Y, W = datasets 

# Perform GCCA
projections_gcca, A_matrices_gcca, G_gcca, time_gcca, mem_gcca = gcca(datasets, k=k)

# Perform MaxVar GCCA
projections_maxvar, A_matrices_maxvar, G_maxvar, time_maxvar, mem_maxvar = maxvar_gcca(datasets, k=k)

# Results analysis
X_proj_gcca, Y_proj_gcca, W_proj_gcca = projections_gcca
X_proj_maxvar, Y_proj_maxvar, W_proj_maxvar = projections_maxvar

print("\n==== GCCA Results ====")
print(f"GCCA Projection Shapes: X: {X_proj_gcca.shape}, Y: {Y_proj_gcca.shape}, W: {W_proj_gcca.shape}")
print(f"GCCA Shared Representation Shape: {G_gcca.shape}")

print("\n==== MaxVar GCCA Results ====")
print(f"MaxVar GCCA Projection Shapes: X: {X_proj_maxvar.shape}, Y: {Y_proj_maxvar.shape}, W: {W_proj_maxvar.shape}")
print(f"MaxVar GCCA Shared Representation Shape: {G_maxvar.shape}")

# Visualize results
plot_gcca_results(G_gcca, X_proj_gcca, Y_proj_gcca, W_proj_gcca)
plot_gcca_results(G_maxvar, X_proj_maxvar, Y_proj_maxvar, W_proj_maxvar)
