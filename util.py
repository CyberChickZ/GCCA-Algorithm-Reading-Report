import numpy as np
import re
from scipy.spatial.distance import cdist

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
    """ 使用 CCA 变换查询最近的法语单词 """
    if word not in en_words:
        return None

    idx = en_words.index(word)
    query_vec = en_vectors[idx].reshape(1, -1)

    # 投影到 CCA 变换后的空间
    query_vec_c = query_vec @ transform_matrix

    # 计算投影后空间的最近邻
    distances = np.linalg.norm(Y_c - query_vec_c, axis=1)
    best_idx = np.argmin(distances)

    return fr_words[best_idx]