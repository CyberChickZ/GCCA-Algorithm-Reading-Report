import numpy as np
import matplotlib.pyplot as plt
from CCA import cca
from util import *

# path （Wiki word vectors, are pre-trained word vectors for languages, trained on Wikipedia using fastText. These vectors in dimension 300 were obtained using the skip-gram model described in Bojanowski et al. (2016) with default parameters.）
en_filepath = "datasets/WordEmbedding/wiki.en.vec"
fr_filepath = "datasets/WordEmbedding/wiki.fr.vec"

# load word vectors
max_words = 20000
en_words, en_vectors = load_word_vectors(en_filepath, max_words=max_words, expected_dim=300)
fr_words, fr_vectors = load_word_vectors(fr_filepath, max_words=max_words, expected_dim=300)

print(f"Loaded {len(en_words)} English words and {len(fr_words)} French words.")

# # Perform CCA on correlated data
n_components = 50  # CCA dimension
X_c, Y_c, A, B, corrs = cca(en_vectors, fr_vectors, n_components)

# Uncorrelated data (English - Random French)
random_fr_vectors = np.random.randn(*fr_vectors.shape)  # Generate completely uncorrelated random French word vectors
X_c1, Y_c1, A1, B1, corrs1 = cca(en_vectors, random_fr_vectors, n_components)

# Result visualization
plt.figure(figsize=(8, 4))
plt.bar(range(1, n_components + 1), corrs, label="Correlated (English-French)")
plt.bar(range(1, n_components + 1), corrs1, label="Uncorrelated (English-Random)", alpha=0.6)
plt.xlabel("CCA Component")
plt.ylabel("Correlation Coefficient")
plt.title("CCA Projection Correlation Comparison")
plt.legend()
plt.show()

# Print the first 3 correlations
print("First 3 correlations of CCA projection (Correlated Data):", corrs[:3])
print("First 3 correlations of CCA projection (Uncorrelated Data):", corrs1[:3])

query_words = ["dog", "cat", "house", "car", "computer"]
print("\n\n**Querying French translations before CCA:**")
for word in query_words:
    translated_before = find_nearest_word(word, en_words, fr_words, en_vectors, fr_vectors)
    print(f"Before CCA: {word} -> {translated_before}")

print("\n\n**Querying French translations after CCA:**")
for word in query_words:
    translated_after = find_nearest_word(word, en_words, fr_words, X_c, Y_c)
    print(f"After CCA: {word} -> {translated_after}")

