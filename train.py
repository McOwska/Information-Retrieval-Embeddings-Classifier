import argparse
import csv
import json
from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def load_data_from_csv(path: str):
    texts: list[str] = []
    labels: list[str] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"CSV must have columns: text,label. Got: {reader.fieldnames}")
        for row in reader:
            texts.append((row.get("text") or "").strip())
            labels.append(str(row.get("label") or "").strip())
    return texts, labels


def load_data_from_hf(repo_id: str, split: str | None):
    try:
        from datasets import load_dataset  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency `datasets`. Install it with: pip install datasets"
        ) from e

    ds_dict = load_dataset(repo_id)
    available_splits = list(ds_dict.keys())
    if not available_splits:
        raise ValueError(f"No splits found for dataset: {repo_id}")

    selected_split = split or ("train" if "train" in ds_dict else available_splits[0])
    ds = ds_dict[selected_split]

    if "text" not in ds.column_names or "label" not in ds.column_names:
        raise ValueError(f"Dataset must have columns: text,label. Got: {ds.column_names}")

    texts = [(t or "").strip() for t in ds["text"]]
    labels = [str(l).strip() for l in ds["label"]]
    return texts, labels, selected_split, available_splits


def parse_args():
    p = argparse.ArgumentParser(description="Train embedding-based text classifier.")
    p.add_argument(
        "--csv",
        default=None,
        help="Path to a local CSV file with columns text,label. If omitted, loads from HF dataset.",
    )
    p.add_argument(
        "--hf-dataset",
        default="McOwska/action-required-text-classification",
        help="Hugging Face dataset repo id (default: McOwska/action-required-text-classification).",
    )
    p.add_argument(
        "--hf-split",
        default=None,
        help="Dataset split to use (default: train if present, else first available).",
    )
    p.add_argument("--test-size", type=float, default=0.33, help="Fraction used for test split.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


args = parse_args()

if args.csv:
    X_all, y_all = load_data_from_csv(args.csv)
    print(f"Loaded {len(X_all)} rows from CSV: {args.csv}")
else:
    X_all, y_all, used_split, splits = load_data_from_hf(args.hf_dataset, args.hf_split)
    print(f"Loaded {len(X_all)} rows from HF dataset: {args.hf_dataset} (split={used_split}, splits={splits})")

X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=args.test_size,
    random_state=args.seed,
    stratify=y_all,
)

print("Test label counts:")
counts: dict[str, int] = {}
for label in y_test:
    counts[label] = counts.get(label, 0) + 1
for label, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
    print(f"  {label}: {count}")

encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(encoder_name)

X_train_emb = embedding_model.encode(list(X_train), show_progress_bar=True)
X_test_emb = embedding_model.encode(list(X_test), show_progress_bar=True)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

results = []
best = {"model_name": None, "macro_f1": float("-inf"), "metrics": None}

for model_name, clf in models.items():
    clf.fit(X_train_emb, y_train)

    y_pred = clf.predict(X_test_emb)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }
    results.append(metrics)

    if macro_f1 > best["macro_f1"]:
        best = {"model_name": model_name, "macro_f1": float(macro_f1), "metrics": metrics}

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)
    print(classification_report(y_test, y_pred))

print("\nSummary (sorted by macro_f1):")
for row in sorted(results, key=lambda r: r["macro_f1"], reverse=True):
    print(
        f'- {row["model"]}: macro_f1={row["macro_f1"]:.4f}, '
        f'weighted_f1={row["weighted_f1"]:.4f}, accuracy={row["accuracy"]:.4f}'
    )

print("\nBest by macro_f1:")
print(best["metrics"])

# Refit the best model on *all* data, then save a bundle you can upload to Hugging Face.
bundle_dir = Path("best_model_bundle")
bundle_dir.mkdir(parents=True, exist_ok=True)

X_all_emb = embedding_model.encode(X_all, show_progress_bar=True)

best_model = models[best["model_name"]]
best_model.fit(X_all_emb, y_all)

joblib.dump(best_model, bundle_dir / "classifier.joblib")
embedding_model.save(str(bundle_dir / "encoder"))

metadata = {
    "encoder_name": encoder_name,
    "best_model_name": best["model_name"],
    "selection_metric": "macro_f1",
    "selection_metrics": best["metrics"],
    "data_source": {"csv": args.csv, "hf_dataset": args.hf_dataset, "hf_split": args.hf_split},
    "files": {
        "classifier": "classifier.joblib",
        "encoder_dir": "encoder/",
        "metadata": "metadata.json",
    },
}
(bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

print(f"\nSaved best model bundle to: {bundle_dir.resolve()}")
