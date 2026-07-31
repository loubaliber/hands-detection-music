"""
train_gesture_cnn.py

Trains a CNN directly on the binarized gesture images.

Why not landmarks (like train_gesture_classifier.py does): MediaPipe's
hand detector is trained on natural RGB images and essentially cannot
find a hand in a black/white silhouette image — on this dataset it
detected a hand in ~2 out of 18,000 images. So for THIS dataset, the
classifier has to work directly on the binarized pixels instead.

Expected dataset layout (same as before):

    dataset_root/
        0/
            img001.png
            ...
        1/
            ...
        ...
        19/
            ...

Usage:
    python train_gesture_cnn.py --dataset dataset/train/train \
        --output gesture_cnn_model.keras
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import layers, models

from gesture_utils import preprocess_for_cnn, CNN_IMAGE_SIZE

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_image_paths(dataset_root: Path):
    for class_dir in sorted((p for p in dataset_root.iterdir() if p.is_dir()),
                             key=lambda p: p.name):
        label = class_dir.name
        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in IMAGE_EXTS:
                yield img_path, label


def load_dataset(dataset_root: Path, max_per_class=None):
    paths = list(iter_image_paths(dataset_root))
    if not paths:
        print(f"ERROR: no images found under {dataset_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(paths)} images across "
          f"{len(set(l for _, l in paths))} classes.")

    X, y = [], []
    per_class_count = {}
    skipped = 0

    for i, (img_path, label) in enumerate(paths):
        if max_per_class is not None and per_class_count.get(label, 0) >= max_per_class:
            continue

        # Images are already binarized -> load as single-channel grayscale.
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue

        X.append(preprocess_for_cnn(img))
        y.append(label)
        per_class_count[label] = per_class_count.get(label, 0) + 1

        if (i + 1) % 1000 == 0:
            print(f"  loaded {i + 1}/{len(paths)}")

    print(f"Loaded {len(X)} usable images, skipped {skipped} unreadable files.")
    print("Per-class counts:", per_class_count)
    return np.array(X, dtype=np.float32), y


def build_model(num_classes: int, img_size: int = CNN_IMAGE_SIZE):
    model = models.Sequential([
        layers.Input(shape=(img_size, img_size, 1)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", default="gesture_cnn_model.keras", type=Path)
    parser.add_argument("--labels-output", default="gesture_cnn_labels.json", type=Path)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: dataset path {args.dataset} does not exist", file=sys.stderr)
        sys.exit(1)

    X, y_labels = load_dataset(args.dataset, args.max_per_class)
    if len(X) == 0:
        print("ERROR: no usable images loaded.", file=sys.stderr)
        sys.exit(1)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    print(f"\nTraining on {len(X_train)} images, validating on {len(X_test)}...")
    model = build_model(num_classes=len(encoder.classes_))
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=4, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in encoder.classes_]))

    model.save(args.output)
    with open(args.labels_output, "w") as f:
        json.dump(list(encoder.classes_), f)

    print(f"\nSaved model to {args.output}")
    print(f"Saved label order to {args.labels_output}")
    print(f"Classes ({len(encoder.classes_)}): {list(encoder.classes_)}")
    print("\nNext: open main.py and fill in GESTURE_LABEL_MAP using these "
          "class names, mapping each to an index 1-10 for the chord/style "
          "tables in music_engine.py.")


if __name__ == "__main__":
    main()
