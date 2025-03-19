import numpy as np
import re
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.decomposition import PCA
import sys

def load_word_vectors(filepath, max_words=20000, expected_dim=300):
    """
    Reads .vec files while handling multiple spaces correctly.
    """
    words = []
    vectors = []

    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()  # Read meta info
        print(f"First line (meta info): {first_line}")  
        next(f)  # Skip meta line

        for i, line in enumerate(f):
            tokens = re.split(r'\s+', line.strip())  # Correct way to split by spaces

            # Ensure correct format: 1 word + 300 values
            if len(tokens) != expected_dim + 1:
                print(f"Skipping malformed line {i}: {line[:50]}...")  
                continue

            word = tokens[0]  # Extract word
            vector = np.array(tokens[1:], dtype=np.float32)  # Convert to float

            words.append(word)
            vectors.append(vector)

            if len(words) >= max_words:
                break  # Limit word count

    return words, np.vstack(vectors)  # Convert list to NumPy array

def compute_average_similarity(source_embeddings, target_embeddings):
    """
    Compute cross-lingual average similarity
    """
    similarities = 1 - cdist(source_embeddings, target_embeddings, metric="cosine")
    return np.mean(np.diag(similarities))

def find_nearest_word(query_word, source_words, target_words, source_embeddings, target_embeddings):
    """
    Finds the nearest cross-lingual word using cosine similarity.
    
    Parameters:
    - query_word: str, the word to query (e.g. "dog")
    - source_words: list[str], the source language word list (e.g. English words)
    - target_words: list[str], the target language word list (e.g. French words)
    - source_embeddings: np.array, the source language word embeddings (N, d)
    - target_embeddings: np.array, the target language word embeddings (N, d)

    Returns:
    - the nearest target language word (str), if `query_word` is not in `source_words`, returns `None`
    """
    if query_word not in source_words:
        print(f"'{query_word}' not found in vocabulary.")
        return None

    # Get the index of query_word in source_words
    idx = source_words.index(query_word)
    query_vec = source_embeddings[idx].reshape(1, -1)

    # Compute cosine similarity (cosine distance lower, similarity higher)
    distances = cdist(query_vec, target_embeddings, metric="cosine")

    # Find the index of the nearest word
    best_idx = np.argmin(distances)
    return target_words[best_idx]

def transform_and_find(word, en_words, fr_words, en_vectors, Y_c, transform_matrix):
    """ Transform query word to CCA space and find nearest French word """
    if word not in en_words:
        return None

    idx = en_words.index(word)
    query_vec = en_vectors[idx].reshape(1, -1)

    # Project to CCA space
    query_vec_c = query_vec @ transform_matrix

    # Compute distances in CCA space
    distances = np.linalg.norm(Y_c - query_vec_c, axis=1)
    best_idx = np.argmin(distances)

    return fr_words[best_idx]

def generate_synthetic_gcca_data(N=1000, I=3, d_list=None, k=None, noise_std=0.1, sparsity_std=0, outliers_noise_scale = 0, result_in_dense=1):
    """
    Generate synthetic GCCA data conforming to the paper's settings
    
    Parameters:
    - N: Sample size (number of samples)
    - I: Number of views (default 3, but can be up to 8)
    - d_list: List of feature dimensions for each view
    - k: Shared latent feature dimension
    - noise_std: Standard deviation of noise

    Returns:
    - X_list: List of sparse view matrices
    """
    
    # Set default values based on paper's setup
    if k is None:
        k = min(60000, int(N * 0.6))  # Ensure k is large enough

    if d_list is None:
        if N < 1000:  # Small-scale test
            d_list = [120, 120, 120]
        else:  # Large-scale test
            d_list = [80000, 80000, 80000]

    # Generate shared latent factor Z (N x k)
    Z = np.random.randn(N, k)

    # Generate mixing matrices A_i (k x d_i) and noise matrices N_i (N x d_i)
    A_list = [np.random.randn(k, d) for d in d_list]
    print("A方差：")
    print(A_list[0].mean(), A_list[0].std())
    N_list = [noise_std * np.random.randn(N, d) for d in d_list]

    # Compute view data X_i = ZA_i + σN_i
    X_list = [Z @ A + N for A, N in zip(A_list, N_list)]

    # Ensure 30% of the features are outliers (noise_scale <0: noise scale based on data energy) (noise_scale > 0 noise scale based on noise_scale)
    if outliers_noise_scale != 0:
        if outliers_noise_scale < 0:
            X_final = [add_outliers(X, noise_scale=None) for X in X_list]
        else:
            X_final = [add_outliers(X, noise_scale=outliers_noise_scale) for X in X_list]

    # Ensure sparsity level is consistent with the paper
    if sparsity_std != 0:
        print(f"Applying sparsity: {sparsity_std:.6f}")  # 确保 sparsity 在 1e-4 ~ 1e-3 之间
        U, _ = X_final
        X_final = [apply_sparsity(X, sparsity_std) for X in X_final]  # 施加正确的稀疏性
        V, _ = X_final
        print("what changed:")
        np.savetxt("matrix_U-V.txt", U-V, fmt="%.4f")
        np.savetxt("matrix_V.txt", V, fmt="%.4f")
        np.savetxt("matrix_U.txt", U, fmt="%.4f")
        
    if result_in_dense == 1:
        return X_final
    else:
        return [sp.csr_matrix(X) for X in X_final]

def add_outliers(X, outlier_feature_ratio=0.3, noise_scale=None):

    X_noisy = X.copy()
    num_samples, num_features = X.shape

    # Select 30% of features as outlier features
    num_outlier_features = int(outlier_feature_ratio * num_features)
    outlier_feature_indices = np.random.choice(num_features, num_outlier_features, replace=False)

    if noise_scale is None:
        # Calculate the standard deviation of the original data as noise intensity
        data_std = np.std(X)  
        noise_scale = data_std  # Match noise energy with the original data

    # Generate noise
    noise = noise_scale * np.random.randn(num_samples, num_outlier_features)

    # Replace selected outlier features
    X_noisy[:, outlier_feature_indices] = noise

    return X_noisy

def apply_sparsity(matrix, sparsity_std):
    """
    Apply strict sparsity to the input matrix:
    - Set exactly `(1 - sparsity_std) × (N × d)` elements to `0`
    - Only retain `sparsity_std × (N × d)` nonzero elements
    - Always return a dense numpy.ndarray
    """

    # Ensure sparsity_std is within the valid range [1e-4, 1e-3]
    if not (1e-4 <= sparsity_std <= 1e-3):
        sparsity_std = np.random.uniform(1e-4, 1e-3)

    # Get matrix shape
    N, d = matrix.shape

    # Compute the total number of nonzero elements needed
    num_nonzero_elements = int(sparsity_std * N * d)

    # Flatten the matrix
    matrix_flat = matrix.flatten()

    # Set all elements to zero initially
    matrix_flat[:] = 0  

    # Randomly select `num_nonzero_elements` positions to remain nonzero
    nonzero_indices = np.random.choice(matrix.size, size=num_nonzero_elements, replace=False)

    # Assign random values back to those positions
    matrix_flat[nonzero_indices] = np.random.randn(num_nonzero_elements)  

    # Reshape back to the original shape
    sparse_matrix = matrix_flat.reshape(N, d)

    return sparse_matrix

def plot_gcca_results(G, X_proj, Y_proj, W_proj):
    """
    Use PCA to project GCCA shared representation G and projected data (X_proj, Y_proj, W_proj) to 2D scatter plots.
    """
    pca = PCA(n_components=2)
    
    # Calculate 2D projections
    G_2D = pca.fit_transform(G)
    X_2D = pca.transform(X_proj)
    Y_2D = pca.transform(Y_proj)
    W_2D = pca.transform(W_proj)

    # Create scatter plots
    plt.figure(figsize=(12, 4))

    # Shared representation G
    plt.subplot(1, 4, 1)
    plt.scatter(G_2D[:, 0], G_2D[:, 1], c=np.arange(G_2D.shape[0]), cmap="viridis", alpha=0.7)
    plt.title("Shared Representation (G)")
    plt.xlabel("PCA Dim 1")
    plt.ylabel("PCA Dim 2")

    # View X
    plt.subplot(1, 4, 2)
    plt.scatter(X_2D[:, 0], X_2D[:, 1], c=np.arange(X_2D.shape[0]), cmap="coolwarm", alpha=0.7)
    plt.title("Projected View X")
    plt.xlabel("PCA Dim 1")
    plt.ylabel("PCA Dim 2")

    # View Y
    plt.subplot(1, 4, 3)
    plt.scatter(Y_2D[:, 0], Y_2D[:, 1], c=np.arange(Y_2D.shape[0]), cmap="plasma", alpha=0.7)
    plt.title("Projected View Y")
    plt.xlabel("PCA Dim 1")
    plt.ylabel("PCA Dim 2")

    # View W
    plt.subplot(1, 4, 4)
    plt.scatter(W_2D[:, 0], W_2D[:, 1], c=np.arange(W_2D.shape[0]), cmap="cividis", alpha=0.7)
    plt.title("Projected View W")
    plt.xlabel("PCA Dim 1")
    plt.ylabel("PCA Dim 2")

    plt.tight_layout()
    plt.show()

from colorist import red

def print_matrix(matrix, name="Matrix"):
    """✅ 仅 0 变成红色，其他数值默认"""
    print(f"{name}:")

    for row in matrix:
        colored_row = [
            red(f"{value:+.2f}") if abs(value) < 1e-10 else f"{value:+.2f}"
            for value in row
        ]
        print(" ".join(colored_row))
    print()
