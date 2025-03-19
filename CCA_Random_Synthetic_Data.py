import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from util import *  
from CoreAlgorithm import *

# Set random seed
np.random.seed(42)

# Set parameters
N = 10000  # Sample size
d1, d2 = 200, 200  # Feature dimensions (already reduced)
k = 100  # Shared latent factor dimension
noise_std = 0.1  # Sparsity

# Generate synthetic data (X, Y, W)
datasets = generate_synthetic_gcca_data(N=N, I=2, d_list=[d1,d2], k=k, noise_std=noise_std)
X, Y = datasets  # Ignore W here

# Convert sparse matrices to dense matrices
X = X.toarray()
Y = Y.toarray()

# Generate unrelated data Y1 (random noise)
Y1 = np.random.randn(N, d2)  # As completely unrelated data

# Perform CCA (100 dimensions)
n_components = 100
X_c, Y_c, A, B, corrs = cca(X, Y, n_components)  # Compute CCA between X and Y
X_c1, Y_c1, A1, B1, corrs1 = cca(X, Y1, n_components)  # Compute CCA between X and Y1 (random data)

# Visualize results
indices = np.arange(1, n_components + 1)
plt.figure(figsize=(8, 4))
plt.bar(indices - 0.2, corrs, width=0.4, label="Correlated Data")
plt.bar(indices + 0.2, corrs1, width=0.4, label="Uncorrelated Data", alpha=0.6)
plt.xlabel("CCA Component")
plt.ylabel("Correlation Coefficient")
plt.title("CCA Projection Correlation Comparison")
plt.legend()
plt.show()

# Print first 3 correlation coefficients
print("First 3 correlations of CCA projection (Correlated Data):", corrs[:3])
print("First 3 correlations of CCA projection (Uncorrelated Data):", corrs1[:3])

