# GAPE_WordEmbedding

A minimal educational repository implementing word embeddings using the
CBOW and Skip-gram architectures. This project builds datasets, trains small
embedding models, and exposes simple similarity evaluation tools so you can
experiment with word representations end-to-end.

**Status:** Example/learning project — small datasets and lightweight models.

## Repository structure

- `data/` — source CSV datasets and vocabulary
	- `cbow_dataset.csv`, `skipgram_dataset.csv`, `vocab.txt`
- `src/` — scripts and utilities
	- `dataset_builder.py` — prepare datasets and vocabulary
	- `train_cbow.py` — train CBOW model
	- `train_skipgram.py` — train Skip-gram model
	- `similarity.py` — compute nearest neighbors / similarity metrics
- `models/` — trained embedding CSV outputs
	- `embeddings_cbow.csv`, `embeddings_skipgram.csv`
- `results/` — training loss logs and similarity outputs
	- `loss_cbow.txt`, `loss_skipgram.txt`, `similarity_results.txt`

## Requirements

Install the project's Python dependencies (recommended in a virtualenv):

```bash
pip install -r requirements.txt
```

## Quick start

1. Prepare datasets (if you change raw data):

```bash
cd src
python dataset_builder.py
```

2. Train a model (from the `src` directory):

```bash
# CBOW
python train_cbow.py

# Skip-gram
python train_skipgram.py
```

Training scripts write embeddings to `models/` and loss logs to `results/`.

3. Compute similarities / evaluate embeddings:

```bash
python similarity.py
```

This will append or write nearest-neighbor comparisons to
`results/similarity_results.txt`.

## Files of interest

- `src/train_cbow.py` and `src/train_skipgram.py` — training loops and
	hyperparameters. Edit these if you want to change learning rate, epochs,
	or embedding dimension.
- `models/embeddings_*.csv` — learned embeddings (one vector per row).
- `results/loss_*.txt` — per-epoch training loss (plain text).

## Notes & next steps

- This repo is intended for experimentation and learning — models are small
	and not optimized for production use.
- You can extend the project by adding validation, batching, or switching to
	PyTorch/TensorFlow for faster training and GPU support.

If you'd like, I can (a) add example CLI flags to the training scripts, (b)
wire up a small notebook to visualize embeddings, or (c) run a training
example and show the output — tell me which option you prefer.
