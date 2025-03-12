import numpy as np
import matplotlib.pyplot as plt

def cca(X, Y, n_components):
    """
    Custom implementation of Canonical Correlation Analysis (CCA) using SVD.
    
    Parameters:
    X: np.array (N, d1) -> First dataset
    Y: np.array (N, d2) -> Second dataset
    n_components: int -> Number of CCA components
    
    Returns:
    X_c, Y_c: Projected data
    A, B: Projection matrices
    corrs: Correlation coefficients
    """
    # Center the data (zero mean)
    X_c = X - np.mean(X, axis=0)
    Y_c = Y - np.mean(Y, axis=0)
    
    # Compute covariance matrices
    Sigma_XX = X_c.T @ X_c
    Sigma_YY = Y_c.T @ Y_c
    Sigma_XY = X_c.T @ Y_c
    
    # Compute pseudo-inverses
    inv_Sigma_XX = np.linalg.pinv(Sigma_XX)
    inv_Sigma_YY = np.linalg.pinv(Sigma_YY)
    
    # Solve the generalized eigenvalue problem
    M = inv_Sigma_XX @ Sigma_XY @ inv_Sigma_YY @ Sigma_XY.T
    eigvals, A = np.linalg.eigh(M)
    A = A[:, ::-1]  # Sort in descending order
    eigvals = eigvals[::-1]
    
    # Select the top components
    A = A[:, :n_components]
    B = inv_Sigma_YY @ Sigma_XY.T @ A
    B /= np.linalg.norm(B, axis=0)  # Normalize B
    
    # Project the data
    X_proj = X_c @ A
    Y_proj = Y_c @ B
    
    # Compute correlation coefficients
    corrs = [np.corrcoef(X_proj[:, i], Y_proj[:, i])[0, 1] for i in range(n_components)]
    
    return X_proj, Y_proj, A, B, corrs

# Generate simulated data
np.random.seed(42)
N = 1000  # Sample size
d1, d2 = 50, 50  # Feature dimensions

# Generate related data
Z = np.random.randn(N, 50)  # Shared latent variables
X = Z @ np.random.randn(50, d1) + 0.1 * np.random.randn(N, d1)  # Random language 1 embedding
Y = Z @ np.random.randn(50, d2) + 0.1 * np.random.randn(N, d2)  # Random language 2 embedding

# Generate uncorrelated data
Z1 = np.random.randn(N, 50)  # Independent latent variables
Y1 = Z1 @ np.random.randn(50, d2) + 0.1 * np.random.randn(N, d2)  # Random language 2 embedding (uncorrelated)

# Perform CCA on correlated data
n_components = 50
X_c, Y_c, A, B, corrs = cca(X, Y, n_components)

# Perform CCA on uncorrelated data
X_c1, Y_c1, A1, B1, corrs1 = cca(X, Y1, n_components)

# Result visualization
plt.figure(figsize=(8, 4))
plt.bar(range(1, n_components + 1), corrs, label='Correlated Data')
plt.bar(range(1, n_components + 1), corrs1, label='Uncorrelated Data', alpha=0.6)
plt.xlabel("CCA Component")
plt.ylabel("Correlation Coefficient")
plt.title("CCA Projection Correlation Comparison")
plt.legend()
plt.show()

# Display the first 3 correlations for both cases
print("First 3 correlations of CCA projection (Correlated Data):", corrs[:3])
print("First 3 correlations of CCA projection (Uncorrelated Data):", corrs1[:3])