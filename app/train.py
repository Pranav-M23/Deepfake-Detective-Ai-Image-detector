import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
    ModelCheckpoint, CSVLogger
)

# ── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE        = 224
BATCH_SIZE      = 64       # larger batch = faster training
TRAIN_DIR       = "dataset/train"
MODEL_PATH      = "model/deepfake_detector.h5"
SAMPLES_PER_CLASS = 8000   # 8K per class = 16K total, fast but effective
EPOCHS_1        = 8        # Phase 1: frozen backbone
EPOCHS_2        = 12       # Phase 2: fine-tune

os.makedirs("model", exist_ok=True)
print("=" * 50)
print("  Deepfake Detector - Training Script")
print("=" * 50)

# ── Data Generators ──────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1,
    brightness_range=[0.85, 1.15],
    validation_split=0.2
)

print("\n[1/5] Loading training data...")
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    seed=42,
    shuffle=True
)

print("[2/5] Loading validation data...")
val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    seed=42,
    shuffle=False
)

# ── Cap samples for speed ─────────────────────────────────────────────────────
total_samples = SAMPLES_PER_CLASS * 2
val_samples   = int(total_samples * 0.2)
train_gen.samples = total_samples
train_gen.n       = total_samples
val_gen.samples   = val_samples
val_gen.n         = val_samples

print(f"\nUsing {total_samples} training samples, {val_samples} validation samples")
print(f"Steps per epoch: {total_samples // BATCH_SIZE}")

# ── Save class indices ────────────────────────────────────────────────────────
print("\n[3/5] Class indices:", train_gen.class_indices)
with open("model/class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)
print("Saved model/class_indices.json")

# ── Build Model ───────────────────────────────────────────────────────────────
print("\n[4/5] Building model...")
base = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base.input, outputs=output)
print(f"Total params: {model.count_params():,}")
print(f"Trainable params (Phase 1): {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# ── Callbacks ─────────────────────────────────────────────────────────────────
def make_callbacks(log_file):
    return [
        EarlyStopping(patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=2, verbose=1, min_lr=1e-7),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
        CSVLogger(log_file, append=True)
    ]

# ── Phase 1: Train head only (frozen backbone) ────────────────────────────────
print("\n" + "=" * 50)
print("  PHASE 1: Training head (backbone frozen)")
print(f"  Epochs: {EPOCHS_1}  |  Batch: {BATCH_SIZE}  |  Samples: {total_samples}")
print("=" * 50)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_1,
    steps_per_epoch=total_samples // BATCH_SIZE,
    validation_steps=val_samples // BATCH_SIZE,
    callbacks=make_callbacks("model/phase1_log.csv"),
    verbose=1
)

p1_acc = max(history1.history['val_accuracy'])
print(f"\nPhase 1 best val accuracy: {p1_acc:.1%}")

# ── Phase 2: Fine-tune top layers of backbone ─────────────────────────────────
print("\n" + "=" * 50)
print("  PHASE 2: Fine-tuning top 20 backbone layers")
print(f"  Epochs: {EPOCHS_2}  |  LR: 1e-5")
print("=" * 50)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"Trainable params (Phase 2): {trainable_count:,}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_2,
    steps_per_epoch=total_samples // BATCH_SIZE,
    validation_steps=val_samples // BATCH_SIZE,
    callbacks=make_callbacks("model/phase2_log.csv"),
    verbose=1
)

p2_acc = max(history2.history['val_accuracy'])
print(f"\nPhase 2 best val accuracy: {p2_acc:.1%}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  TRAINING COMPLETE")
print(f"  Phase 1 best accuracy : {p1_acc:.1%}")
print(f"  Phase 2 best accuracy : {p2_acc:.1%}")
print(f"  Model saved to        : {MODEL_PATH}")
print(f"  Logs saved to         : model/phase1_log.csv")
print(f"                          model/phase2_log.csv")
print("=" * 50)