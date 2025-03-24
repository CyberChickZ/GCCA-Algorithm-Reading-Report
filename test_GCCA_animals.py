import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tracemalloc
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from CoreAlgorithm import gcca, maxvar_gcca, altmaxvar_gcca

# Set data paths
DATASET_PATH = "datasets/newdata/embeddings"
ANIMALS = ["bird", "cat", "chicken", "cow", "dog", "frog", "lion", "monkey", "pig", "sheep"]

# Load embeddings
def load_embeddings():
    """Load image and sound embeddings, ensuring correct shape alignment."""
    img_embeddings, sound_embeddings = [], []

    for animal in ANIMALS:
        img_path = os.path.join(DATASET_PATH, f"{animal}_img.npy")
        sound_path = os.path.join(DATASET_PATH, f"{animal}_sound.npy")

        if not os.path.exists(img_path) or not os.path.exists(sound_path):
            print(f"⚠️ Missing data for {animal}: {img_path} or {sound_path}")
            continue  # Skip missing animal data

        img_data = np.load(img_path)
        sound_data = np.load(sound_path)

        if img_data.shape[0] != 10 or sound_data.shape[0] != 10:
            print(f"⚠️ Shape mismatch for {animal}: img={img_data.shape}, sound={sound_data.shape}")

        img_embeddings.append(img_data)
        sound_embeddings.append(sound_data)

    # Stack embeddings to ensure proper format
    img_embeddings = np.vstack(img_embeddings)
    sound_embeddings = np.vstack(sound_embeddings)

    print(f"\n✅ Loaded embeddings: img={img_embeddings.shape}, sound={sound_embeddings.shape}")
    return img_embeddings, sound_embeddings

# Compute cosine similarity matrix
def cosine_similarity_matrix(X):
    """Compute cosine similarity using optimized vectorized method."""
    sim_matrix = 1 - cdist(X, X, metric="cosine")  # Faster computation
    return sim_matrix

# Run GCCA variants with time and memory tracking
def run_gcca_methods(data_views, k=50):
    """Run GCCA, MaxVar-GCCA, and AltMaxVar-GCCA and return results with time and memory analysis."""
    print("Running GCCA methods...")
    results = {}
    time_records = {}
    memory_records = {}

    def execute_method(method_func, method_name):
        """Measure time and memory usage for a GCCA method."""
        tracemalloc.start()
        start_time = time.time()

        _, _, G, _, _ = method_func(data_views, k=k)

        elapsed_time = time.time() - start_time
        current, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results[method_name] = G
        time_records[method_name] = elapsed_time
        memory_records[method_name] = peak_memory / (1024 ** 2)  # Convert to MB

        print(f"[✓] {method_name} completed in {elapsed_time:.3f} sec | Peak Memory: {memory_records[method_name]:.2f} MB")

    # Run all three GCCA methods
    execute_method(gcca, "GCCA")
    execute_method(maxvar_gcca, "MaxVar-GCCA")
    execute_method(lambda x, k: altmaxvar_gcca(x, k, constraint="nonneg"), "AltMaxVar-GCCA")

    return results, time_records, memory_records

# Visualize cosine similarity
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_similarity_matrix(sim_matrix, title):
    """Plot cosine similarity matrix with improved readability and correct axis alignment."""
    fig, ax = plt.subplots(figsize=(12, 10))  # Adjust figure size

    # Ensure labels are correctly spaced and ordered
    step = 10  # Show every 10th label
    num_samples = sim_matrix.shape[0]

    # Generate correct category labels
    y_labels = []
    for i in range(len(ANIMALS)):
        y_labels.extend([ANIMALS[i]] * 10)

    # Apply heatmap with optimized color mapping
    sns.heatmap(sim_matrix, cmap="magma", vmin=0, vmax=1, robust=True,
                xticklabels=y_labels, yticklabels=y_labels, ax=ax)

    # Fix label positions and orientation
    ax.set_xticks(np.arange(0, num_samples, step))
    ax.set_yticks(np.arange(0, num_samples, step))
    ax.set_xticklabels(y_labels[::step], rotation=45, fontsize=10)
    ax.set_yticklabels(y_labels[::step], fontsize=10)
    
    # Ensure correct alignment of x and y labels
    ax.invert_yaxis()  # Match x-axis order with y-axis

    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Samples")
    plt.colorbar(ax.collections[0])  # Add colorbar for reference
    plt.show()


# Visualize t-SNE results
def plot_tsne(G_matrix, method_name):
    """Plot t-SNE visualization of GCCA projected embeddings."""
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    transformed = tsne.fit_transform(G_matrix)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap("tab10", len(ANIMALS))

    for i, animal in enumerate(ANIMALS):
        idx = range(i * 10, (i + 1) * 10)
        plt.scatter(transformed[idx, 0], transformed[idx, 1], color=colors(i), label=animal)

    plt.legend()
    plt.title(f"t-SNE Projection ({method_name})")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.show()

# Plot execution time and memory usage
def plot_performance(time_records, memory_records):
    """Plot execution time and memory usage comparison for different GCCA methods."""
    methods = list(time_records.keys())
    execution_times = list(time_records.values())
    memory_usage = list(memory_records.values())

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("GCCA Method")
    ax1.set_ylabel("Execution Time (sec)", color="tab:blue")
    ax1.bar(methods, execution_times, color="tab:blue", alpha=0.6, label="Execution Time")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Memory Usage (MB)", color="tab:red")
    ax2.plot(methods, memory_usage, color="tab:red", marker="o", label="Memory Usage")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    plt.title("Performance Comparison: Time & Memory")
    plt.show()

# Main program
if __name__ == "__main__":
    print("Loading embeddings...")
    img_embeddings, sound_embeddings = load_embeddings()

    # Create 2-view dataset
    data_views = [img_embeddings, sound_embeddings]

    # Run GCCA variants
    results, time_records, memory_records = run_gcca_methods(data_views, k=50)

    # Compute and visualize cosine similarity
    for method, G_matrix in results.items():
        print(f"\nComputing Cosine Similarity for {method}...")
        sim_matrix = cosine_similarity_matrix(G_matrix)
        plot_similarity_matrix(sim_matrix, title=f"Cosine Similarity - {method}")

        # t-SNE visualization
        plot_tsne(G_matrix, method_name=method)

    # Plot time & memory performance
    plot_performance(time_records, memory_records)

    print("\nAll tests completed!")
