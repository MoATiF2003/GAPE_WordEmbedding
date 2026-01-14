import csv
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def load_vocab(vocab_path):
    word_to_id = {}
    id_to_word = {}

    with open(vocab_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["word"]
            idx = int(row["id"])
            word_to_id[word] = idx
            id_to_word[idx] = word

    return word_to_id, id_to_word


word_to_id, id_to_word = load_vocab("../data/vocab.txt")
vocab_size = len(word_to_id)


def one_hot(index, size):
    vec = np.zeros(size)
    vec[index] = 1.0
    return vec



X, y = [], []

with open("../data/skipgram_dataset.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        X.append(one_hot(word_to_id[row["input_word"]], vocab_size))
        y.append(one_hot(word_to_id[row["context_word"]], vocab_size))

X = np.array(X)
y = np.array(y)


embedding_dim = 10

model = keras.Sequential([
    layers.Dense(embedding_dim, input_shape=(vocab_size,), activation="linear"),
    layers.Dense(vocab_size, activation="softmax")
])

optimizer = keras.optimizers.Adam(learning_rate=0.003)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
)

history = model.fit(
    X,
    y,
    epochs=400,
    batch_size=4,
    verbose=1
)


with open("../results/loss_skipgram.txt", "w") as f:
    for loss in history.history["loss"]:
        f.write(f"{loss}\n")


embedding_matrix = model.layers[0].get_weights()[0]

with open("../models/embeddings_skipgram.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["word"] + [f"dim_{i}" for i in range(embedding_dim)])
    for idx, word in id_to_word.items():
        writer.writerow([word] + embedding_matrix[idx].tolist())
