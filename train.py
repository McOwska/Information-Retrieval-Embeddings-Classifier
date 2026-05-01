import pandas as pd
from sentence_transformers import SentenceTransformer

import json
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

df = pd.read_csv("dataset_hard_100.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.33,
    random_state=42,
    stratify=df["label"]
)

print(y_test.value_counts())

encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(encoder_name)

X_train_emb = embedding_model.encode(X_train.tolist(), show_progress_bar=True)
X_test_emb = embedding_model.encode(X_test.tolist(), show_progress_bar=True)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
}

results = []
best = {"model_name": None, "macro_f1": float("-inf"), "clf": None, "metrics": None}

for model_name, clf in models.items():
    clf.fit(X_train_emb, y_train)

    y_pred = clf.predict(X_test_emb)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    metrics = {
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
    results.append(metrics)

    if macro_f1 > best["macro_f1"]:
        best = {
            "model_name": model_name,
            "macro_f1": macro_f1,
            "clf": clf,
            "metrics": metrics,
        }

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)
    print(classification_report(y_test, y_pred))

results_df = pd.DataFrame(results)
print("\nSummary:")
print(results_df.sort_values(by="macro_f1", ascending=False))

print("\nBest by macro_f1:")
print(best["metrics"])

# Refit the best model on *all* data, then save a bundle you can upload to Hugging Face.
bundle_dir = Path("best_model_bundle")
bundle_dir.mkdir(parents=True, exist_ok=True)

X_all = df["text"].tolist()
y_all = df["label"]
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
    "files": {
        "classifier": "classifier.joblib",
        "encoder_dir": "encoder/",
        "metadata": "metadata.json",
    },
}
(bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

print(f"\nSaved best model bundle to: {bundle_dir.resolve()}")
