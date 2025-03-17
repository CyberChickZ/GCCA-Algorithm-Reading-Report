import numpy as np

def cca(X, Y, n_components):
    """
    Canonical Correlation Analysis (CCA) using SVD-based optimization.
    
    Parameters:
    X: np.array (N, d1) -> First dataset (samples × features)
    Y: np.array (N, d2) -> Second dataset (samples × features)
    n_components: int -> Number of CCA components
    
    Returns:
    X_proj, Y_proj: Projected data
    A, B: Projection matrices
    corrs: Correlation coefficients
    """
    N = X.shape[0]  # 样本数
    
    # Center the data (zero mean)
    X_c = X - np.mean(X, axis=0)
    Y_c = Y - np.mean(Y, axis=0)
    
    # Compute covariance matrices (unbiased estimation)
    Sigma_XX = (X_c.T @ X_c) / (N - 1)
    Sigma_YY = (Y_c.T @ Y_c) / (N - 1)
    Sigma_XY = (X_c.T @ Y_c) / (N - 1)
    
    # SVD for stability
    Ux, sx, Vx = np.linalg.svd(Sigma_XX)
    Uy, sy, Vy = np.linalg.svd(Sigma_YY)
    
    inv_Sigma_XX = Vx.T @ np.diag(1 / sx) @ Vx
    inv_Sigma_YY = Vy.T @ np.diag(1 / sy) @ Vy

    # Solve the generalized eigenvalue problem
    M = inv_Sigma_XX @ Sigma_XY @ inv_Sigma_YY @ Sigma_XY.T
    eigvals, A = np.linalg.eigh(M)
    
    # Sort eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    A = A[:, idx[:n_components]]
    
    # Compute B from A
    B = inv_Sigma_YY @ Sigma_XY.T @ A
    B /= np.linalg.norm(B, axis=0, keepdims=True)  # Normalize B
    
    # Normalize X_c and Y_c
    X_c /= np.std(X_c, axis=0, keepdims=True)
    Y_c /= np.std(Y_c, axis=0, keepdims=True)

    # Project the data
    X_proj = X_c @ A
    Y_proj = Y_c @ B

    # Compute correlation coefficients
    corrs = [np.corrcoef(X_proj[:, i], Y_proj[:, i])[0, 1] for i in range(n_components)]
    
    return X_proj, Y_proj, A, B, corrs
