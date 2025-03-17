import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from CCA import cca  # 你的自定义 CCA 实现

# Set random seed
np.random.seed(42)

# Set parameters
N = 10000  # Sample size
d1, d2 = 200, 300  # Feature dimensions

# Generate shared latent variables
Z = np.random.randn(N, 100)  # 100-dimensional shared features
X = Z @ np.random.randn(100, d1) + 0.1 * np.random.randn(N, d1)  # Related X
Y = Z @ np.random.randn(100, d2) + 0.1 * np.random.randn(N, d2)  # Related Y

# Reduce dimensionality of Y to match X (200 dimensions)
pca_Y = PCA(n_components=200)
Y_200 = pca_Y.fit_transform(Y)  # Reduce dimensionality of Y

# Generate unrelated Y1 (true random noise)
Y1 = np.random.randn(N, d2)  # Completely unrelated Y1 data

# Number of CCA components
n_components = 100

# --- Perform CCA using your custom implementation ---
X_c, Y_c, A, B, corrs = cca(X, Y_200, n_components)  # CCA on related data
X_c1, Y_c1, A1, B1, corrs1 = cca(X, Y1, n_components)  # CCA on unrelated data

# --- Perform CCA using scikit-learn ---
cca_sklearn = CCA(n_components=n_components)
X_sklearn, Y_sklearn = cca_sklearn.fit_transform(X, Y_200)  # Fit on correlated data
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
