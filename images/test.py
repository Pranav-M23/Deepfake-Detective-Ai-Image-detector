# test_model.py - paste this, replacing everything in test.py
import json
import numpy as np
import os

print("Loading libraries...")
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

print("Loading model...")
model = load_model(r"S:\Projects\Ai Video Detector\model\deepfake_detector.h5")
print("Model loaded!")

with open(r"S:\Projects\Ai Video Detector\model\class_indices.json") as f:
    class_indices = json.load(f)

print("Class indices:", class_indices)
idx_to_class = {v: k for k, v in class_indices.items()}

# Auto-test all images in current folder
current_dir = os.path.dirname(os.path.abspath(__file__))
image_extensions = ('.jpg', '.jpeg', '.png', '.webp')

found = False
for fname in os.listdir(current_dir):
    if fname.lower().endswith(image_extensions):
        found = True
        img_path = os.path.join(current_dir, fname)
        img = image.load_img(img_path, target_size=(224, 224))
        arr = image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        prob = model.predict(arr, verbose=0)[0][0]
        predicted_idx = 1 if prob > 0.5 else 0
        label = idx_to_class[predicted_idx]
        confidence = prob if predicted_idx == 1 else 1 - prob
        print(f"{fname:40s} → {label} ({confidence:.1%})")

if not found:
    print("No images found in", current_dir)
    print("Drop some .jpg or .png images into the images/ folder and run again")