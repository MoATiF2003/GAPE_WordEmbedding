import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(path):
    words = []
    vectors = []

    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            words.append(row[0])
            vectors.append([float(x) for x in row[1:]])

    return words, np.array(vectors)

def top_k_similar(query_word, words, vectors, k=5):
    if query_word not in words:
        return []

    idx = words.index(query_word)
    query_vec = vectors[idx].reshape(1, -1)

    sims = cosine_similarity(query_vec, vectors)[0]
    ranked = sorted(
        [(words[i], sims[i]) for i in range(len(words)) if words[i] != query_word],
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:k]

# Change path if you want Skip-gram instead
embedding_path = "../models/embeddings_cbow.csv"

words, vectors = load_embeddings(embedding_path)

query_words = [
    "network",
    "bandwidth",
    "latency",
    "deployment",
    "calls"
]

with open("../results/similarity_results.txt", "w") as f:
    for query in query_words:
        f.write(f"Query word: {query}\n")
        results = top_k_similar(query, words, vectors)

        for word, score in results:
            f.write(f"  {word}: {score:.4f}\n")

        f.write("\n")

