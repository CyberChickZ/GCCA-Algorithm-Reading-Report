import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from scipy.spatial.distance import cdist
from util import load_word_vectors, find_nearest_word, transform_and_find
from CCA import cca  # Your CCA implementation

# Load word vectors
# en_filepath = "datasets/WordEmbedding/wiki.en.vec"
# fr_filepath = "datasets/WordEmbedding/wiki.es.vec"
en_filepath = "datasets/WordEmbedding/not work/wiki.multi.en.vec"
fr_filepath = "datasets/WordEmbedding/not work/wiki.multi.fr.vec"
max_words = 20000

en_words, en_vectors = load_word_vectors(en_filepath, max_words=max_words, expected_dim=300)
fr_words, fr_vectors = load_word_vectors(fr_filepath, max_words=max_words, expected_dim=300)

print(f"Loaded {len(en_words)} English words and {len(fr_words)} French words.")

# Define CCA parameters
n_components = 50  # Number of CCA dimensions

# from sklearn.preprocessing import StandardScaler
# # Standardize data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(en_vectors)
# Y_scaled = scaler.fit_transform(fr_vectors)

# Run Your CCA
X_c, Y_c, A, B, corrs_custom = cca(en_vectors, fr_vectors, n_components)

# Run Sklearn CCA for comparison
cca_sklearn = CCA(n_components=n_components, max_iter=200000)
X_sklearn, Y_sklearn = cca_sklearn.fit_transform(en_vectors, fr_vectors)

# Compare Correlation Coefficients
corrs_sklearn = [np.corrcoef(X_sklearn[:, i], Y_sklearn[:, i])[0, 1] for i in range(n_components)]

# Visualization
plt.figure(figsize=(8, 4))
plt.plot(range(1, n_components + 1), corrs_custom, label="Your CCA", marker='o')
plt.plot(range(1, n_components + 1), corrs_sklearn, label="Sklearn CCA", marker='x')
plt.xlabel("CCA Component")
plt.ylabel("Correlation Coefficient")
plt.title("CCA Projection Correlation Comparison")
plt.legend()
plt.show()
print(f"Sum of top 50 correlations (Your CCA): {sum(corrs_custom[:n_components])}")
print(f"Sum of top 50 correlations (Sklearn CCA): {sum(corrs_sklearn[:n_components])}")

# Compare Translations Before and After CCA
query_words = ["dog", "cat", "house", "car", "computer"]

print("\n Querying French translations before CCA:")
for word in query_words:
    translated_before = find_nearest_word(word, en_words, fr_words, en_vectors, fr_vectors)
    print(f"Before CCA: {word} -> {translated_before}")

print("\n Querying French translations after Your CCA:")
for word in query_words:
    translated_after = transform_and_find(word, en_words, fr_words, en_vectors, Y_c, A)
    print(f"Your CCA: {word} -> {translated_after}")

print("\n Querying French translations after Sklearn CCA:")
for word in query_words:
    translated_after_sklearn = transform_and_find(word, en_words, fr_words, en_vectors, Y_sklearn, cca_sklearn.x_weights_)
    print(f"Sklearn CCA: {word} -> {translated_after_sklearn}")

print("Your CCA A matrix first row:", A[0, :5])
print("Sklearn CCA weights first row:", cca_sklearn.x_weights_[0, :5])