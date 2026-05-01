import pandas as pd
from sentence_transformers import SentenceTransformer

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

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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

for model_name, clf in models.items():
    clf.fit(X_train_emb, y_train)

    y_pred = clf.predict(X_test_emb)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    })

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)
    print(classification_report(y_test, y_pred))

results_df = pd.DataFrame(results)
print("\nSummary:")
print(results_df.sort_values(by="macro_f1", ascending=False))