"""Apple disease classifier web app.

Upload a leaf photo -> model predicts scab / black rot / cedar rust / healthy.
Two models: field-adapted (default, real-world photos) and baseline (lab photos).
Railway-ready: respects $PORT.
"""
import io
import os
from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, render_template, request
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

BASE = Path(__file__).resolve().parent
MODELS = {
    "adapted": ("apple_field_adapted.pt", "Field-adapted (real-world photos)"),
    "baseline": ("apple_baseline.pt", "Baseline (clean lab photos)"),
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB uploads

print("Loading backbone...", flush=True)
weights = EfficientNet_B0_Weights.DEFAULT
backbone = efficientnet_b0(weights=weights)
backbone.classifier = nn.Identity()
backbone.eval()
tfm = weights.transforms()

heads = {}
for key, (fname, label) in MODELS.items():
    ckpt = torch.load(BASE / fname, map_location="cpu", weights_only=False)
    head = nn.Linear(1280, len(ckpt["class_names"]))
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    heads[key] = (head, ckpt["class_names"], label)
print("Models ready.", flush=True)


@torch.no_grad()
def classify(pil_img, which):
    head, class_names, _ = heads[which]
    x = tfm(pil_img).unsqueeze(0)
    f = backbone.features(x)
    f = torch.flatten(backbone.avgpool(f), 1)
    probs = torch.softmax(head(f), dim=1)[0]
    scored = sorted(
        ((class_names[i], float(probs[i])) for i in range(len(class_names))),
        key=lambda t: t[1], reverse=True,
    )
    return scored


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", models=MODELS, result=None, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    which = request.form.get("model", "adapted")
    if which not in heads:
        which = "adapted"
    f = request.files.get("photo")
    if not f or not f.filename:
        return render_template("index.html", models=MODELS, result=None,
                               error="Please choose a photo first.")
    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
    except Exception:
        return render_template("index.html", models=MODELS, result=None,
                               error="That file is not a readable image.")
    scored = classify(img, which)
    top_name, top_prob = scored[0]
    result = {
        "top": top_name,
        "confidence": f"{top_prob:.1%}",
        "bars": [(n, f"{p:.1%}", p * 100) for n, p in scored],
        "model_label": heads[which][2],
    }
    return render_template("index.html", models=MODELS, result=result,
                           error=None, active_model=which)


@app.route("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
