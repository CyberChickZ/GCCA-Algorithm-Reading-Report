import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from util import *
from CoreAlgorithm import *

# Set parameters
N = 1000  # Sample size
d1, d2 = 200, 200  # Feature dimensions (already reduced)
k = 100  # Shared latent factor dimension
noise_std = 1  # noise cofactor
outliers_noise_scale = -1 # active 30% outliers and scale is based on data enery

# Generate synthetic data (X, Y)
datasets = generate_synthetic_gcca_data(N=N, I=2, d_list=[d1,d2], k=k, noise_std=noise_std, outliers_noise_scale=outliers_noise_scale)
X, Y = datasets

# Generate unrelated data Y1 (random noise)
Y1 = np.random.randn(N, d2)  # As completely unrelated data

# --- Perform CCA using your custom implementation ---
n_components = 80
X_c, Y_c, A, B, corrs = cca(X, Y, n_components)  # CCA on related data
X_c1, Y_c1, A1, B1, corrs1 = cca(X, Y1, n_components)  # CCA on unrelated data

# --- Perform CCA using scikit-learn ---
cca_sklearn = CCA(n_components=n_components, max_iter=20000)
X_sklearn, Y_sklearn = cca_sklearn.fit_transform(X, Y)  # Fit on correlated data
X_sklearn1, Y_sklearn1 = cca_sklearn.fit_transform(X, Y1)  # Fit on uncorrelated data

# Compute correlation coefficients for sklearn CCA
corrs_sklearn = [np.corrcoef(X_sklearn[:, i], Y_sklearn[:, i])[0, 1] for i in range(n_components)]
corrs_sklearn1 = [np.corrcoef(X_sklearn1[:, i], Y_sklearn1[:, i])[0, 1] for i in range(n_components)]

# --- Plot Comparison ---
indices = np.arange(1, n_components + 1)
plt.figure(figsize=(10, 5))

plt.bar(indices - 0.3, corrs, width=0.3, label="Custom CCA (Correlated Data)", alpha=0.8)
plt.bar(indices, corrs_sklearn, width=0.3, label="Sklearn CCA (Correlated Data)", alpha=0.8)
plt.bar(indices + 0.3, corrs1, width=0.3, label="Custom CCA (Uncorrelated Data)", alpha=0.6)
plt.bar(indices + 0.6, corrs_sklearn1, width=0.3, label="Sklearn CCA (Uncorrelated Data)", alpha=0.6)

plt.xlabel("CCA Component")
plt.ylabel("Correlation Coefficient")
plt.title("CCA Projection Correlation Comparison: Custom vs. Sklearn")
plt.legend()
plt.show()

# Show first 3 correlations for comparison
print("First 3 correlations of Custom CCA (Correlated Data):", corrs[:3])
print("First 3 correlations of Sklearn CCA (Correlated Data):", corrs_sklearn[:3])
print("First 3 correlations of Custom CCA (Uncorrelated Data):", corrs1[:3])
print("First 3 correlations of Sklearn CCA (Uncorrelated Data):", corrs_sklearn1[:3])
