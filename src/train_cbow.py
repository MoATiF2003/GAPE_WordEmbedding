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

X = []
y = []

with open("../data/cbow_dataset.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        context_words = row["context_words"].split()
        target_word = row["target_word"]

        context_vec = np.zeros(vocab_size)
        for w in context_words:
            context_vec += one_hot(word_to_id[w], vocab_size)

        context_vec = context_vec / len(context_words)  # average
        X.append(context_vec)

        y.append(one_hot(word_to_id[target_word], vocab_size))

X = np.array(X)
y = np.array(y)

embedding_dim = 10

model = keras.Sequential([
    layers.Dense(embedding_dim, input_shape=(vocab_size,), activation="linear"),
    layers.Dense(vocab_size, activation="softmax")
])

learning_rate = 0.05  
optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy"
)

epochs = 100
history = model.fit(X, y, epochs=epochs, verbose=1)

with open("../results/loss_cbow.txt", "w") as f:
    for loss in history.history["loss"]:
        f.write(f"{loss}\n")


# Extract embeddings (weights of first Dense layer)
embedding_matrix = model.layers[0].get_weights()[0]

with open("../models/embeddings_cbow.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["word"] + [f"dim_{i}" for i in range(embedding_matrix.shape[1])])

    for idx, word in id_to_word.items():
        writer.writerow([word] + embedding_matrix[idx].tolist())
