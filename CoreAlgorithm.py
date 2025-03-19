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

def gcca(datasets, n_components=50, tol=1e-6, max_iter=2000):
    num_views = len(datasets)
    N = datasets[0].shape[0]

    centered_datasets = [X - np.mean(X, axis=0) for X in datasets]
    A_matrices = [np.random.randn(X.shape[1], n_components) for X in datasets]
    G = np.random.randn(N, n_components)

    norms = []

    # Start timing and memory measurement
    tracemalloc.start()
    start_time = time.process_time()  # Only measure process time, excluding external influences

    for iteration in range(max_iter):
        prev_G = G.copy()

        # Update A_i
        for i in range(num_views):
            X_i = centered_datasets[i]
            A_matrices[i] = np.linalg.pinv(X_i) @ G

        # Update G
        sum_matrices = sum(X @ A for X, A in zip(centered_datasets, A_matrices))
        G = sum_matrices / num_views

        # Calculate the change in G
        norm_diff = np.linalg.norm(G - prev_G)
        norms.append(norm_diff)

        # Calculate progress and display it
        progress = (iteration + 1) / max_iter * 100
        sys.stdout.write(f"\rIteration {iteration+1}/{max_iter} - Progress: {progress:.2f}% - ΔG: {norm_diff:.6e}")
        sys.stdout.flush()

        if norm_diff < tol:
            print(f"\nGCCA converged in {iteration+1} iterations\n")
            break

    # Stop timing and memory measurement
    end_time = time.process_time()
    memory_usage = tracemalloc.get_traced_memory()  # Get memory usage
    tracemalloc.stop()

    total_time = end_time - start_time
    max_memory = memory_usage[1] / (1024 * 1024)  # Convert to MB

    print(f"\nGCCA Execution Time: {total_time:.4f} seconds")
    print(f"GCCA Peak Memory Usage: {max_memory:.2f} MB")

    plt.ion
    # Plot the convergence curve
    plt.figure(figsize=(6, 4))
    plt.plot(norms, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Change in G")
    plt.title("GCCA Convergence Plot")
    plt.show()

    projections = [X @ A for X, A in zip(centered_datasets, A_matrices)]
    return projections, A_matrices, G
