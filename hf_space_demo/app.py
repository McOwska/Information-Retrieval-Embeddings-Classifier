import os
from pathlib import Path

import gradio as gr
import joblib
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer


MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "McOwska/action-requirement-classifier")
REVISION = os.getenv("MODEL_REVISION")  # optional


def _load_bundle():
    local_dir = snapshot_download(
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        revision=REVISION,
        local_dir=None,
    )
    base = Path(local_dir)
    encoder_dir = base / "encoder"
    clf_path = base / "classifier.joblib"

    encoder = SentenceTransformer(str(encoder_dir))
    clf = joblib.load(clf_path)
    return encoder, clf


ENCODER, CLF = _load_bundle()


def predict(text: str):
    text = (text or "").strip()
    if not text:
        return "Please enter some text."
    emb = ENCODER.encode([text])
    pred = CLF.predict(emb)[0]
    return str(pred)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=6, label="Text"),
    outputs=gr.Label(label="Predicted label"),
    title="Action Requirement Classifier",
    description=f"Loads model artifacts from `{MODEL_REPO_ID}`.",
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch()

