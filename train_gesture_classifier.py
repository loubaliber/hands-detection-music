"""
train_gesture_classifier.py

Trains a gesture classifier on the Kaggle Hand Gesture Recognition
dataset using MediaPipe hand landmarks as features (NOT raw pixels).

Why landmarks instead of a CNN on raw images:
  - MediaPipe already gives a 21-point hand skeleton per frame at
    real-time speed, so re-using those points for classification
    means the live app does ONE model pass (detection) instead of
    two, which is what keeps latency low.
  - A landmark-based classifier (RandomForest / MLP) trains in
    seconds to minutes on CPU and runs inference in <1ms, versus a
    CNN which needs a GPU to hit the FPS target comfortably.

Expected dataset layout (standard Kaggle image-classification style):

    dataset_root/
        gesture_class_1/
            img001.jpg
            img002.jpg
            ...
        gesture_class_2/
            ...
        ...

Each subfolder name is used as the class label. If your download has
an extra nesting level (e.g. subject folders), point --dataset at the
folder that directly contains the class subfolders, or adjust
`iter_image_paths` below.

Usage:
    python train_gesture_classifier.py --dataset /path/to/dataset \
        --output gesture_model.pkl
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions

from gesture_utils import landmarks_to_features

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_HAND_MODEL = "hand_landmarker.task"


def iter_image_paths(dataset_root: Path):
    """Yields (image_path, class_label) for every image in the dataset."""
    for class_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        label = class_dir.name
        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in IMAGE_EXTS:
                yield img_path, label


def extract_dataset_features(dataset_root: Path, hand_model_path: Path,
                              max_per_class: int | None = None):
    """
    Runs MediaPipe's HandLandmarker (Tasks API, IMAGE mode) over every
    training image and returns (X, y) where X is an (N, 63) feature
    matrix and y is a list of class label strings. Images where no
    hand is detected are skipped and reported.
    """
    options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    X, y = [], []
    skipped = 0
    per_class_count = {}

    paths = list(iter_image_paths(dataset_root))
    if not paths:
        print(f"ERROR: no images found under {dataset_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(paths)} candidate images across "
          f"{len(set(l for _, l in paths))} classes.")

    for i, (img_path, label) in enumerate(paths):
        if max_per_class is not None and per_class_count.get(label, 0) >= max_per_class:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)

        if not result.hand_landmarks:
            skipped += 1
            continue

        features = landmarks_to_features(result.hand_landmarks[0])
        X.append(features)
        y.append(label)
        per_class_count[label] = per_class_count.get(label, 0) + 1

        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(paths)} images "
                  f"({skipped} skipped so far)")

    landmarker.close()

    print(f"Done. Usable samples: {len(X)}, skipped (no hand found): {skipped}")
    print("Per-class counts:", per_class_count)

    return np.array(X, dtype=np.float32), y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path,
                         help="Path to dataset root (folder of class subfolders)")
    parser.add_argument("--output", default="gesture_model.pkl", type=Path,
                         help="Where to save the trained model bundle")
    parser.add_argument("--max-per-class", type=int, default=None,
                         help="Optional cap on images per class (speeds up iteration)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--hand-model", default=DEFAULT_HAND_MODEL, type=Path,
                         help="Path to MediaPipe hand_landmarker.task model file")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: dataset path {args.dataset} does not exist", file=sys.stderr)
        sys.exit(1)

    if not args.hand_model.exists():
        print(f"ERROR: hand landmark model not found at {args.hand_model}\n"
              f"Download it with:\n"
              f"  curl -L -o {args.hand_model} "
              f"https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              f"hand_landmarker/float16/1/hand_landmarker.task",
              file=sys.stderr)
        sys.exit(1)

    X, y_labels = extract_dataset_features(args.dataset, args.hand_model, args.max_per_class)

    if len(X) == 0:
        print("ERROR: no usable samples extracted. Check dataset path/structure.",
              file=sys.stderr)
        sys.exit(1)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    print(f"\nTraining RandomForestClassifier "
          f"(n_estimators={args.n_estimators}) on {len(X_train)} samples...")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    bundle = {
        "model": clf,
        "label_encoder": encoder,
        "feature_dim": X.shape[1],
    }
    joblib.dump(bundle, args.output)
    print(f"\nSaved model bundle to {args.output}")
    print(f"Classes ({len(encoder.classes_)}): {list(encoder.classes_)}")
    print("\nNext step: open music_engine.py and map each of these class "
          "names to a chord (left hand) / performance style (right hand) "
          "in GESTURE_TO_CHORD / GESTURE_TO_STYLE.")


if __name__ == "__main__":
    main()
