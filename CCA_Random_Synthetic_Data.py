import matplotlib.pyplot as plt
import numpy as np
from CCA import *

# Set random seed
np.random.seed(42)

# Set parameters
N = 10000  # Sample size
d1, d2 = 200, 300  # Feature dimensions

# Generate shared latent variables
Z = np.random.randn(N, 100)  # 50-dimensional shared features
X = Z @ np.random.randn(100, d1) + 0.1 * np.random.randn(N, d1)  # Related X
Y = Z @ np.random.randn(100, d2) + 0.1 * np.random.randn(N, d2)  # Related Y

# Reduce dimensionality of Y to match X (200 dimensions)
from sklearn.decomposition import PCA
pca_Y = PCA(n_components=200)
Y_200 = pca_Y.fit_transform(Y)  # Reduce dimensionality of Y

# Generate unrelated Y1 (true random noise)
Y1 = np.random.randn(N, d2)  # Completely unrelated Y1 data

# Perform CCA (100 dimensions)
n_components = 100
X_c, Y_c, A, B, corrs = cca(X, Y_200, n_components)  # CCA on related data
X_c1, Y_c1, A1, B1, corrs1 = cca(X, Y1, n_components)  # CCA on unrelated data

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

# Show first 3 correlations
print("First 3 correlations of CCA projection (Correlated Data):", corrs[:3])
print("First 3 correlations of CCA projection (Uncorrelated Data):", corrs1[:3])