import matplotlib.pyplot as plt
import numpy as np
from GCCA import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from util import generate_synthetic_gcca_data
from util import plot_gcca_results

# Set random seed
np.random.seed(42)

# Set parameters
N = 150  # Sample size
d1, d2, d3 = 120, 120, 120  # Feature dimensions for X, Y, and W
k = 60  # Shared latent factor dimension
sparsity = 0.3  # Sparsity

# Generate synthetic data (X, Y, W)
datasets = generate_synthetic_gcca_data(N=N, d1=d1, d2=d2, d3=d3, k=k, noise_std=0.1, sparsity=sparsity)
X, Y, W = datasets  # Ignore W here

# Convert sparse matrices to dense matrices
X = X.toarray()
Y = Y.toarray()
W = W.toarray()

# perform GCCA
projections, A_matrices, G = gcca_convergence_plot([X, Y, W], n_components=50)

# results analysis
X_proj, Y_proj, W_proj = projections
print(f"GCCA Projection Shapes: X: {X_proj.shape}, Y: {Y_proj.shape}, W: {W_proj.shape}")
print(f"Shared Representation Shape: {G.shape}")

# visualize 
plot_gcca_results(G, X_proj, Y_proj, W_proj)