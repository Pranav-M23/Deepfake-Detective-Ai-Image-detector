import os

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from app.model import load_model
from app.video_utils import sample_frames

# ---------------------------------------------------------------------------
# Setup (lazy — model loaded on first call so this module is importable cheaply)
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.getenv("MODEL_PATH", "model/deepfake_model.pt")
MODEL_TYPE = os.getenv("MODEL_TYPE", "efficientnet")

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH, str(DEVICE), model_type=MODEL_TYPE)
    return _model


# MUST match training transform (224x224)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Class labels — index order must match training dataset.classes
CLASS_NAMES = ["FAKE", "REAL"]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_image(image_path: str) -> tuple[str, float]:
    """Predict a single image from disk. Returns (label, confidence%)."""
    pil_img = Image.open(image_path).convert("RGB")
    return _predict_pil(pil_img)


def predict_pil(pil_img: Image.Image) -> tuple[str, float]:
    """Predict a PIL Image already loaded in memory."""
    return _predict_pil(pil_img)


def predict_video(
    video_path: str,
    every_n: int = 15,
    max_frames: int = 20,
) -> dict:
    """
    Sample frames from a video file and return an aggregated verdict.

    Returns a dict with keys: verdict, fake_votes, real_votes,
    frames_analyzed, avg_fake_confidence, frame_details.
    """
    frames = sample_frames(video_path, every_n=every_n, max_frames=max_frames)
    if not frames:
        return {"error": "No frames could be extracted from the video."}

    frame_results = []
    fake_confs, real_confs = [], []

    for i, frame_bgr in enumerate(frames):
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        label, conf = _predict_pil(pil_img)
        frame_results.append({"frame_index": i, "label": label, "confidence": conf})
        if label == "FAKE":
            fake_confs.append(conf)
            real_confs.append(100 - conf)
        else:
            real_confs.append(conf)
            fake_confs.append(100 - conf)

    fake_votes = sum(1 for r in frame_results if r["label"] == "FAKE")
    real_votes = len(frame_results) - fake_votes
    verdict = "FAKE" if fake_votes >= real_votes else "REAL"

    return {
        "verdict": verdict,
        "fake_votes": fake_votes,
        "real_votes": real_votes,
        "frames_analyzed": len(frame_results),
        "avg_fake_confidence": round(sum(fake_confs) / len(fake_confs), 2),
        "avg_real_confidence": round(sum(real_confs) / len(real_confs), 2),
        "frame_details": frame_results,
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _predict_pil(pil_img: Image.Image) -> tuple[str, float]:
    mdl = _get_model()
    tensor = transform(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = mdl(tensor)
        probs = F.softmax(outputs, dim=1)
        conf_val, predicted = torch.max(probs, 1)
    label = CLASS_NAMES[predicted.item()]
    return label, round(conf_val.item() * 100, 2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else input("Enter image or video path: ")
    ext = os.path.splitext(path)[1].lower()

    if ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        result = predict_video(path)
        print(f"Verdict      : {result['verdict']}")
        print(f"Fake votes   : {result['fake_votes']} / {result['frames_analyzed']}")
        print(f"Avg fake conf: {result['avg_fake_confidence']:.2f}%")
    else:
        label, conf = predict_image(path)
        print(f"Prediction : {label}")
        print(f"Confidence : {conf:.2f}%")
