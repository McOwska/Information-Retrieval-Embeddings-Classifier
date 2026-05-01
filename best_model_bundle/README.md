# Action Requirement Classifier

This repository contains a lightweight text classifier for predicting whether a message is:

- `action_required`
- `informational`
- `optional_action`

## What’s inside

- `classifier.joblib`: a scikit-learn classifier trained on SentenceTransformer embeddings
- `encoder/`: the SentenceTransformer encoder saved via `SentenceTransformer.save()`
- `metadata.json`: bundle metadata (model selection metric, encoder name, etc.)

## Quickstart (Python)

```python
import joblib
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("./encoder")
clf = joblib.load("classifier.joblib")

texts = ["Please upload a clearer photo of your ID."]
emb = encoder.encode(texts)
pred = clf.predict(emb)
print(pred[0])
```

## Notes

- The classifier expects embeddings from the included `encoder/` directory.
- If you move files around, update paths accordingly.

