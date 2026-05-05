# Information Retrieval — Embeddings Classifier

A small text classification project built with SentenceTransformers embeddings + a lightweight scikit-learn classifier.

It predicts one of:

- `action_required`
- `informational`
- `optional_action`

## Repository contents

- `train.py` — trains multiple classifiers on sentence embeddings, selects the best by macro-F1, and saves a bundle
- `dataset_hard_100.csv` — legacy/local CSV (same schema) used earlier
- `dataset_hard_300.csv`, `dataset_300.csv`, `instruction_action_dataset.csv` — additional datasets
- `hf_space_demo/` — Hugging Face Space (Gradio) demo that loads the model from the Hub

## How it works

1. Encode `text` with `sentence-transformers/all-MiniLM-L6-v2`
2. Train a classifier on the embeddings (LogReg / LinearSVC / RandomForest)
3. Select the best model on the test split using macro-F1
4. Refit that best model on all data and save a portable bundle (classifier + encoder)

## Train locally

Create a virtualenv, install deps, then:

```bash
python train.py
```

By default, `train.py` reads the dataset directly from Hugging Face:

- Dataset: `McOwska/action-required-text-classification` (split `train`)

If you prefer a local CSV:

```bash
python train.py --csv dataset_hard_100.csv
```

This creates a local folder `best_model_bundle/` containing:

- `classifier.joblib`
- `encoder/` (SentenceTransformer files)
- `metadata.json`

Note: `best_model_bundle/` is intentionally ignored by git (large artifacts). The uploaded version lives on Hugging Face.

## Hugging Face

- Model repo: `McOwska/action-requirement-classifier`
- Demo Space: `McOwska/action-requirement-classifier-demo`

