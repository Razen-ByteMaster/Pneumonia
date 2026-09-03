# 🫁 Pneumonia Chest X-Ray Classifier

> Upload a chest X-ray — get back **Normal** or **Pneumonia** with a confidence score.
> A fine-tuned ResNet50 behind a tiny Flask service, trained end-to-end in the included notebook. 🩻

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-API-black?logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-yellow)
![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen)

## ✨ What is this?

A deep-learning image classifier that detects pneumonia in chest X-ray photos. Send an X-ray to one endpoint — get back a prediction with confidence. The full training pipeline (Colab notebook, 25 epochs) ships in the repo, so the result is reproducible, not magic.

| Capability | Details |
|---|---|
| 🧠 Model | ResNet50, fine-tuned for binary classification (`Normal` / `Pneumonia`) |
| 🖼️ Input | Any common image format (JPEG/PNG) — auto-resized, center-cropped to 224×224, ImageNet-normalized |
| 📊 Output | Predicted class + softmax confidence (0–1 fraction) |
| ⚡ Device | CUDA when available, CPU otherwise — no code change needed |
| 🚀 API | Single `POST /predict` endpoint, JSON in / JSON out |

> 📈 **Test accuracy: XX.X%** — replace with your measured number before publishing.

## ⚙️ How it works

```
Upload (JPEG/PNG)
      ↓
Resize 256 → CenterCrop 224×224 → ToTensor → Normalize (ImageNet mean/std)
      ↓
ResNet50 (custom 2-class head, Pneumonia.pth weights)
      ↓
Softmax → argmax → { "prediction": "Pneumonia", "confidence": 0.9742 }
```

## 🚀 Quickstart

```bash
cd Pneumonia
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python app.py
```

> ⚠️ The trained weights file `Pneumonia.pth` (~94 MB) must sit next to `app.py`. See [Model weights](#-model-weights) before pushing to GitHub.

Test it (server runs at http://127.0.0.1:5000 by default):

```bash
curl -X POST http://127.0.0.1:5000/predict -F "file=@xray_image.jpeg"
```

## 🔍 Sample output

```json
{
  "prediction": "Pneumonia",
  "confidence": 0.9742
}
```

Low confidence? Treat it like a lab assistant raising its hand — confirm with a second test, don't blindly trust the number. ⚠️

## 🏋️ Training

The full pipeline lives in `Pneumonia (1).ipynb` (Google Colab, 22 cells):

1. Downloads the chest X-ray dataset from Kaggle
2. Loads ResNet50 pretrained, freezes all layers, unfreezes `layer3+` for fine-tuning
3. Handles class imbalance with weighted CrossEntropyLoss
4. Trains with AdamW + ReduceLROnPlateau for 25 epochs
5. Evaluates on the test split and saves `Pneumonia.pth`

- Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## 🛠️ Tech stack

| Layer | Tech | Why |
|---|---|---|
| Model | ResNet50 (torchvision) | Strong transfer-learning baseline for medical images |
| Framework | PyTorch | Training + inference in one ecosystem |
| API | Flask | Minimal serving layer, zero frontend needed |
| Preprocessing | torchvision transforms + Pillow | Resize, crop, tensor, ImageNet normalize |
| Training | Colab + AdamW + weighted loss | GPU training with imbalance handling |

## 🏋️ Model weights

- `Pneumonia.pth` is a full ResNet50 state dict (~94 MB) — just under GitHub's 100 MB per-file limit.
- Recommended: keep it **out of git** (uncomment the `*.pth` line in `.gitignore`) and attach it to a **GitHub Release** instead; link the release in this section.
- At runtime the file must be named exactly `Pneumonia.pth` and sit next to `app.py`.

## 📁 Project structure

```
Pneumonia/
├── app.py                  # Flask app: model load + POST /predict
├── Pneumonia (1).ipynb     # Colab training notebook (22 cells)
├── Pneumonia.pth           # Fine-tuned ResNet50 weights (~94 MB, see above)
├── requirements.txt        # Python dependencies
└── README.md               # You are here 👋
```

## 🗺️ Roadmap

- [ ] `GET /health` endpoint (model loaded? device?)
- [ ] Confidence as percentage (match the Malaria API's 0–100 style)
- [ ] Input validation (reject non-images with a clean 400)
- [ ] Confidence-threshold flag (warn when unsure)
- [ ] Batch prediction endpoint
- [ ] Dockerfile + pinned `torch` CUDA/CPU builds
- [ ] Grad-CAM heatmaps (show *where* the opacity is)

## 📄 License

MIT. Built by [Razen Moamen](https://github.com/Razen-ByteMaster) as a portfolio project. PRs welcome! 🎉

> ⚕️ **Disclaimer:** educational/portfolio project — not a medical device. Never use it for real diagnosis.
