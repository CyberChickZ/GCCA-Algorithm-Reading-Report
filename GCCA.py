import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import tracemalloc  # Measure memory usage

def gcca(datasets, n_components=50, tol=1e-4, max_iter=1000):
    """
    Generalized Canonical Correlation Analysis (GCCA)

    Parameters:
    - datasets: list of np.array [(N, d1), (N, d2), (N, d3), ...]
      Each dataset corresponds to a different view.
    - n_components: int, Number of GCCA components
    - tol: float, Convergence tolerance
    - max_iter: int, Maximum number of iterations

    Returns:
    - projections: list of np.array, projected data for each dataset
    - transformation_matrices: list of np.array, transformation matrices A_i
    - shared_representation: np.array, the shared representation G
    """
    
    num_views = len(datasets)  # Number of datasets
    N = datasets[0].shape[0]  # Number of samples

    # Center the data (zero mean)
    centered_datasets = [X - np.mean(X, axis=0) for X in datasets]

    # Initialize projection matrices A_i
    A_matrices = [np.random.randn(X.shape[1], n_components) for X in datasets]

    # Initialize G (shared representation)
    G = np.random.randn(N, n_components)

    for iteration in range(max_iter):
        prev_G = G.copy()

        # Step 1: Update each A_i
        for i in range(num_views):
            X_i = centered_datasets[i]
            A_matrices[i] = np.linalg.pinv(X_i) @ G

        # Step 2: Update shared representation G
        sum_matrices = sum(X @ A for X, A in zip(centered_datasets, A_matrices))
        G = sum_matrices / num_views

        # Step 3: Check convergence
        if np.linalg.norm(G - prev_G) < tol:
            print(f"GCCA converged in {iteration} iterations")
            break

    # Compute final projected data
    projections = [X @ A for X, A in zip(centered_datasets, A_matrices)]

    return projections, A_matrices, G

def gcca_convergence_plot(datasets, n_components=50, tol=1e-6, max_iter=2000):
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

    # Plot the convergence curve
    plt.figure(figsize=(6, 4))
    plt.plot(norms, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Change in G")
    plt.title("GCCA Convergence Plot")
    plt.show()

    projections = [X @ A for X, A in zip(centered_datasets, A_matrices)]
    return projections, A_matrices, G
