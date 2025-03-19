import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import tracemalloc  # Measure memory usage

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
    N = X.shape[0]  # Sample Size
    
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

import numpy as np
import time
import tracemalloc
import sys

def gcca(datasets, k, max_iter=1000, tol=1e-6):
    """
    Standard GCCA (Residual Minimization) with Time & Memory Profiling.

    Parameters:
        datasets (list of np.ndarray): List of datasets, each of shape (N, d_i).
        k (int): Number of canonical components.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        projections (list of np.ndarray): Projected datasets.
        A_matrices (list of np.ndarray): Transformation matrices.
        G (np.ndarray): Common representation.
        total_time (float): Total execution time in seconds.
        max_memory (float): Peak memory usage in MB.
    """
    # Input validation
    if not all(X.shape[0] == datasets[0].shape[0] for X in datasets):
        raise ValueError("All datasets must have the same number of samples (N).")

    I = len(datasets)  # Number of datasets
    N = datasets[0].shape[0]  # Number of samples

    # Initialize A_matrices and G
    A_matrices = [np.random.randn(X.shape[1], k) for X in datasets]
    G = np.random.randn(N, k)

    # Start memory and time tracing
    tracemalloc.start()
    start_time = time.time()
    print("\n")
    for iteration in range(max_iter):
        G_old = G.copy()

        # Update A_matrices
        for i in range(I):
            X = datasets[i]
            A_matrices[i] = np.linalg.lstsq(X, G, rcond=None)[0]

        # Update G
        G = np.mean([X @ A for X, A in zip(datasets, A_matrices)], axis=0)

        # Calculate convergence
        convergence = np.linalg.norm(G - G_old, ord='fro')

        # Progress display
        progress = (iteration + 1) / max_iter * 100
        sys.stdout.write(
            f"\rGCCA Iteration {iteration + 1}/{max_iter} - Progress: {progress:.2f}% - ΔG: {convergence:.6f}"
        )
        sys.stdout.flush()

        # Check convergence
        if convergence < tol:
            print(f"\nConvergence reached in {iteration} iteration!\n")
            break

    # Calculate projections
    projections = [X @ A for X, A in zip(datasets, A_matrices)]

    # Measure time and memory
    end_time = time.time()
    memory_usage = tracemalloc.get_traced_memory()
    total_time = end_time - start_time
    max_memory = memory_usage[1] / (1024 * 1024)  # Convert to MB

    # Print final results
    print(f"\nGCCA Execution Time: {total_time:.4f} seconds")
    print(f"GCCA Peak Memory Usage: {max_memory:.2f} MB")

    return projections, A_matrices, G, total_time, max_memory

def maxvar_gcca(datasets, k):
    """
    MaxVar GCCA Algorithm with Time & Memory Profiling
    """
    N = datasets[0].shape[0]
    I = len(datasets)

    tracemalloc.start()
    start_time = time.time()

    M = np.zeros((N, N))

    for X in datasets:
        X_pinv = np.linalg.pinv(X)
        M += X @ X_pinv  

    U, _, _ = np.linalg.svd(M)
    G_opt = U[:, :k]  

    A_matrices = [np.linalg.pinv(X) @ G_opt for X in datasets]
    projections = [X @ A for X, A in zip(datasets, A_matrices)]

    end_time = time.time()
    memory_usage = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_time = end_time - start_time
    max_memory = memory_usage[1] / (1024 * 1024)  

    print(f"\nMaxVar GCCA Execution Time: {total_time:.4f} seconds")
    print(f"MaxVar GCCA Peak Memory Usage: {max_memory:.2f} MB")

    return projections, A_matrices, G_opt, total_time, max_memory

def prox_operator(matrix, constraint='none', param=0.1):
    """
    Proximal operator for constraints (Non-Negativity / Sparsity)
    
    :param matrix: Input matrix
    :param constraint: 'none' (default), 'nonneg' (Non-Negative), 'sparse' (L1 regularization)
    :param param: Regularization parameter (for sparsity)
    :return: Processed matrix
    """
    if constraint == 'nonneg':  
        return np.maximum(0, matrix)  # Enforce non-negativity
    elif constraint == 'sparse':  
        return np.sign(matrix) * np.maximum(0, np.abs(matrix) - param)  # Soft thresholding
    else:
        return matrix  # No constraint

def altmaxvar_gcca(datasets, k, max_iter=50, inner_iter=10, alpha=0.01, gamma=0.5, tol=1e-6, constraint='none', param=0.1):
    """
    AltMaxVar GCCA Algorithm with Time & Memory Profiling

    :param datasets: List of view matrices [X1, X2, ..., XI]
    :param k: Target shared representation dimension
    :param max_iter: Outer loop max iterations
    :param inner_iter: Inner loop iterations for Q_i updates
    :param alpha: Learning rate for Q_i updates
    :param gamma: Mixing parameter for G update
    :param tol: Convergence tolerance
    :param constraint: Constraint type ('none', 'nonneg', 'sparse')
    :param param: Regularization parameter for constraints
    :return: projections, A_matrices, G, execution time, memory usage
    """
    I = len(datasets)  # Number of views
    N = datasets[0].shape[0]  # Number of samples

    # Start tracking time and memory
    tracemalloc.start()
    start_time = time.process_time()

    # Initialize projection matrices Q_i and shared representation G
    A_matrices = [np.random.randn(X.shape[1], k) for X in datasets]
    G = np.random.randn(N, k)

    for r in range(max_iter):
        # Step 1: Optimize Q_i iteratively using gradient updates
        for t in range(inner_iter):
            for i in range(I):
                X = datasets[i]
                gradient = -X.T @ (X @ A_matrices[i] - G) / N  # Compute gradient
                H = A_matrices[i] - alpha * gradient  # Gradient descent update
                A_matrices[i] = prox_operator(H, constraint=constraint, param=param)  # Apply constraint

        # Step 2: Update shared representation G
        R = gamma * sum(X @ A for X, A in zip(datasets, A_matrices)) / I + (1 - gamma) * G
        U, _, Vt = np.linalg.svd(R, full_matrices=False)
        G_new = U @ Vt  # SVD-based update

        # Check convergence
        if np.linalg.norm(G - G_new, ord='fro') < tol:
            break
        G = G_new

    # Compute final projections
    projections = [X @ A for X, A in zip(datasets, A_matrices)]

    # Stop tracking time and memory
    end_time = time.process_time()
    memory_usage = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_time = end_time - start_time
    max_memory = memory_usage[1] / (1024 * 1024)  # Convert to MB

    print(f"\nAltMaxVar GCCA Execution Time: {total_time:.4f} seconds")
    print(f"AltMaxVar GCCA Peak Memory Usage: {max_memory:.2f} MB")

    return projections, A_matrices, G, total_time, max_memory