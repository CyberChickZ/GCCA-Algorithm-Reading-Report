import numpy as np
import re
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.decomposition import PCA

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



def generate_synthetic_gcca_data(N=1000, d1=100, d2=100, d3=150, k=50, noise_std=0.1, sparsity=0.3):
    """
    Generate synthetic GCCA data conforming to the paper's settings
    
    Parameters:
    - N: Sample size
    - d1, d2, d3: Feature dimensions of three views
    - k: Shared latent factor dimension (size of common low-dimensional space)
    - noise_std: Noise standard deviation
    - sparsity: Sparsity level (0~1), controls non-zero element ratio
    
    Returns:
    - datasets: [X, Y, W] three-view data matrices (sparse representation)
    """
    
    # Generate shared latent factor Z (N x k)
    Z = np.random.randn(N, k)

    # Generate mixing matrices A_i (k x d_i)
    A1 = np.random.randn(k, d1)
    A2 = np.random.randn(k, d2)
    A3 = np.random.randn(k, d3)

    # Generate noise matrices N_i (N x d_i)
    N1 = noise_std * np.random.randn(N, d1)
    N2 = noise_std * np.random.randn(N, d2)
    N3 = noise_std * np.random.randn(N, d3)

    # Compute view data X_i = ZA_i + σN_i
    X = Z @ A1 + N1  # (N x d1)
    Y = Z @ A2 + N2  # (N x d2)
    W = Z @ A3 + N3  # (N x d3)

    # Apply sparsity using scipy.sparse
    def apply_sparsity(matrix, sparsity):
        mask = np.random.rand(*matrix.shape) < sparsity  # Generate sparse mask
        sparse_matrix = sp.csr_matrix(matrix * mask)  # Convert to efficient CSR sparse format
        return sparse_matrix

    X_sparse = apply_sparsity(X, sparsity)
    Y_sparse = apply_sparsity(Y, sparsity)
    W_sparse = apply_sparsity(W, sparsity)

    assert sp.issparse(X_sparse), "X_sparse is not a sparse matrix!"
    assert sp.issparse(Y_sparse), "Y_sparse is not a sparse matrix!"
    assert sp.issparse(W_sparse), "W_sparse is not a sparse matrix!"

   # visual sparsity matrix
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.spy(X_sparse, markersize=0.5)
    plt.title("X (View 1)")

    plt.subplot(1, 3, 2)
    plt.spy(Y_sparse, markersize=0.5)
    plt.title("Y (View 2)")

    plt.subplot(1, 3, 3)
    plt.spy(W_sparse, markersize=0.5)
    plt.title("W (View 3)")

    plt.tight_layout()
    plt.show()

    return [X_sparse, Y_sparse, W_sparse]

# # Generate synthetic data
# datasets = generate_synthetic_gcca_data()

# # View data shape and sparsity
# for i, data in enumerate(datasets):
#     sparsity_ratio = data.nnz / (data.shape[0] * data.shape[1])  # Compute non-zero element ratio
#     print(f"View {i+1} shape: {data.shape}, Nonzero elements: {sparsity_ratio:.4f}")

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