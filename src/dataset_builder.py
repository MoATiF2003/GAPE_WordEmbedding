import re
import csv
from collections import Counter

corpus = """
A telecom network optimises coverage by tuning antenna parameters and managing
interference between neighbouring cells. Bandwidth allocation supports voice calls,
video streaming, and data sessions during peak hours. Fault management monitors
dropped calls and latency spikes through network analytics. Upgrades involve fibre
backhaul, 5G deployment, and careful rollout planning.
"""

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    return tokens

tokens = preprocess_text(corpus)

word_freq = Counter(tokens)
vocab = {word: idx for idx, word in enumerate(word_freq.keys())}

with open("../data/vocab.txt", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["word", "id", "frequency"])
    for word, idx in vocab.items():
        writer.writerow([word, idx, word_freq[word]])

WINDOW_SIZE = 4

cbow_data = []
skipgram_data = []

for i in range(len(tokens)):
    target = tokens[i]
    start = max(0, i - WINDOW_SIZE)
    end = min(len(tokens), i + WINDOW_SIZE + 1)

    context = [tokens[j] for j in range(start, end) if j != i]

    # CBOW
    if len(context) > 0:
        cbow_data.append((context, target))

    # Skip-gram
    for ctx_word in context:
        skipgram_data.append((target, ctx_word))

with open("../data/cbow_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["context_words", "target_word"])
    for context, target in cbow_data:
        writer.writerow([" ".join(context), target])

with open("../data/skipgram_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["input_word", "context_word"])
    for input_word, context_word in skipgram_data:
        writer.writerow([input_word, context_word])

