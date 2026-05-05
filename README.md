# Information Retrieval — Embeddings Classifier

A small text classification project built with SentenceTransformers embeddings + a lightweight scikit-learn classifier.

It predicts one of:

- `action_required`
- `informational`
- `optional_action`

## Repository contents

- `train.py` — trains multiple classifiers on sentence embeddings, selects the best by macro-F1, and saves a bundle
- `dataset_hard_100.csv` — 100-row dataset used for the current demo/model
- `dataset_hard_300.csv`, `dataset_300.csv`, `instruction_action_dataset.csv` — additional datasets
- `hf_space_demo/` — Hugging Face Space (Gradio) demo that loads the model from the Hub

## How it works

1. Encode `text` with `sentence-transformers/all-MiniLM-L6-v2`
2. Train a classifier on the embeddings (LogReg / LinearSVC / RandomForest)
3. Select the best model on the test split using macro-F1
4. Refit that best model on all data and save a portable bundle (classifier + encoder)

## Train locally

Create a virtualenv, then:

```bash
python train.py
```

This creates a local folder `best_model_bundle/` containing:

- `classifier.joblib`
- `encoder/` (SentenceTransformer files)
- `metadata.json`

Note: `best_model_bundle/` is intentionally ignored by git (large artifacts). The uploaded version lives on Hugging Face.

## Hugging Face

- Model repo: `McOwska/action-requirement-classifier`
- Demo Space: `McOwska/action-requirement-classifier-demo`

## Quick inference (from a local bundle)

```python
import joblib
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("best_model_bundle/encoder")
clf = joblib.load("best_model_bundle/classifier.joblib")

texts = ["Please upload a clearer photo of your ID."]
emb = encoder.encode(texts)
print(clf.predict(emb)[0])
```

