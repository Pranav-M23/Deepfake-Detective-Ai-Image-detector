import kagglehub
import shutil
import os

# Download
path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
print("Downloaded to:", path)

# Check what's inside
for root, dirs, files in os.walk(path):
    level = root.replace(path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level < 2:  # only show top 2 levels
        for f in files[:3]:
            print(f'{indent}  {f}')

splits = ["train", "test"]
classes = ["REAL", "FAKE"]

project_train = r"S:\Projects\Ai Video Detector\dataset\train"

for split in splits:
    for cls in classes:
        src = os.path.join(path, split, cls)
        dst = os.path.join(project_train, cls)
        os.makedirs(dst, exist_ok=True)
        
        copied = 0
        for img in os.listdir(src):
            shutil.copy(os.path.join(src, img), os.path.join(dst, img))
            copied += 1
        print(f"Copied {copied} images from {split}/{cls} → {dst}")

print("\nFinal counts:")
for cls in classes:
    count = len(os.listdir(os.path.join(project_train, cls)))
    print(f"  {cls}: {count} images")