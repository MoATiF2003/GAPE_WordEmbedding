import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(path):
    words, vectors = [], []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            words.append(row[0])
            vectors.append([float(x) for x in row[1:]])
    return words, np.array(vectors)

def top_k(query, words, vectors, k=5):
    idx = words.index(query)
    qv = vectors[idx].reshape(1, -1)
    sims = cosine_similarity(qv, vectors)[0]
    ranked = sorted(
        [(words[i], sims[i]) for i in range(len(words)) if words[i] != query],
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:k]

# Load both embeddings
words_cbow, vecs_cbow = load_embeddings("../models/embeddings_cbow.csv")
words_sg, vecs_sg = load_embeddings("../models/embeddings_skipgram.csv")

query_words = ["network", "bandwidth", "latency", "deployment", "calls"]

with open("../results/similarity_results.txt", "w") as f:
    for q in query_words:
        f.write(f"Query word: {q}\n")

        f.write("CBOW:\n")
        for w, s in top_k(q, words_cbow, vecs_cbow):
            f.write(f"  {w}: {s:.4f}\n")

        f.write("Skip-gram:\n")
        for w, s in top_k(q, words_sg, vecs_sg):
            f.write(f"  {w}: {s:.4f}\n")

        f.write("\n")
