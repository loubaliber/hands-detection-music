"""
main.py

Real-time two-hand gesture musical instrument.

Pipeline per frame:
  Webcam -> MediaPipe HandLandmarker (2 hands, for location + handedness
  only) -> crop each hand region -> binarize to match training data ->
  CNN classifier (gesture id + confidence) -> per-hand GestureStabilizer
  (temporal smoothing) -> MusicEngine.set_state() -> MIDI output, while
  an OpenCV window shows landmarks + live status.

Run:
    python main.py --model gesture_cnn_model.keras --labels gesture_cnn_labels.json
Quit:
    press 'q' in the video window.
"""

import argparse
import json
import os
import time

# Import music_engine (and therefore pygame.midi / SDL2) BEFORE cv2.
# Both cv2 and pygame bundle their own copy of SDL2 on macOS (see the
# "Class ... implemented in both ...libSDL2..." warnings at startup) —
# whichever loads first wins the Objective-C runtime symbol resolution.
# Importing pygame first avoids cv2's copy silently breaking pygame's
# audio/MIDI internals.
from music_engine import MusicEngine, GESTURE_TO_CHORD, GESTURE_TO_STYLE

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions
import numpy as np
import tensorflow as tf

from gesture_utils import (
    GestureStabilizer, HAND_CONNECTIONS, landmarks_bbox,
    binarize_hand_crop, preprocess_for_cnn,
)

DEFAULT_HAND_MODEL = "hand_landmarker.task"

# ---------------------------------------------------------------------------
# EDIT ME after training: map your dataset's actual class label strings
# (printed by train_gesture_cnn.py, e.g. "0".."19") to the gesture indices
# (1-10) used by GESTURE_TO_CHORD / GESTURE_TO_STYLE in music_engine.py.
# If left empty, classes are auto-assigned 1..N in sorted order (fine for
# a first smoke test, arbitrary chords otherwise).
# ---------------------------------------------------------------------------
GESTURE_LABEL_MAP = {
    # "0": 1,
    # "1": 2,
    # "2": 3,
    # ...
}

CONFIDENCE_THRESHOLD = 0.55


def build_label_map(classes):
    if GESTURE_LABEL_MAP:
        return dict(GESTURE_LABEL_MAP)
    # GESTURE_TO_CHORD / GESTURE_TO_STYLE only define indices 1-10, so if
    # the dataset has more than 10 classes, wrap around rather than
    # producing indices that silently match nothing. This is a fallback
    # for smoke-testing only -- fill in GESTURE_LABEL_MAP deliberately
    # to pick which 10 of your classes actually get used.
    sorted_classes = sorted(classes)
    if len(sorted_classes) > 10:
        print(f"WARNING: {len(sorted_classes)} classes found but only 10 "
              f"gesture slots exist (GESTURE_TO_CHORD/GESTURE_TO_STYLE). "
              f"Auto-wrapping indices 1-10 -- fill in GESTURE_LABEL_MAP "
              f"to control this deliberately.")
    return {label: (i % 10) + 1 for i, label in enumerate(sorted_classes)}


def classify_hand_crop(model, class_labels, label_map, bgr_crop):
    """Returns (gesture_index or None, confidence)."""
    if bgr_crop.size == 0:
        return None, 0.0
    binary = binarize_hand_crop(bgr_crop)
    features = preprocess_for_cnn(binary)[np.newaxis, ...]  # add batch dim
    probs = model.predict(features, verbose=0)[0]
    best_idx = int(np.argmax(probs))
    confidence = float(probs[best_idx])
    label = class_labels[best_idx]
    gesture_index = label_map.get(label)
    if gesture_index is None or confidence < CONFIDENCE_THRESHOLD:
        return None, confidence
    return gesture_index, confidence


def draw_landmarks(frame, hand_landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 120, 255), -1)


def draw_ui(frame, fps, left_info, right_info, chord_name, style_name, status):
    h, w = frame.shape[:2]
    overlay_lines = [
        f"FPS: {fps:.1f}",
        f"Status: {status}",
        f"Left hand (harmony):  {left_info}",
        f"Right hand (style):   {right_info}",
        f"Chord: {chord_name or '-'}",
        f"Style: {style_name or '-'}",
    ]
    y = 25
    for line in overlay_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 120), 1, cv2.LINE_AA)
        y += 25
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gesture_cnn_model.keras",
                         help="Path to trained CNN model from train_gesture_cnn.py")
    parser.add_argument("--labels", default="gesture_cnn_labels.json",
                         help="Path to label list saved by train_gesture_cnn.py")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--smoothing-window", type=int, default=8,
                         help="Frames of majority-vote smoothing per hand")
    parser.add_argument("--hand-model", default=DEFAULT_HAND_MODEL,
                         help="Path to MediaPipe hand_landmarker.task model file")
    parser.add_argument("--midi-device", type=int, default=None,
                         help="MIDI output device id (see --list-midi-devices)")
    parser.add_argument("--list-midi-devices", action="store_true",
                         help="Print available MIDI devices and exit")
    args = parser.parse_args()

    if args.list_midi_devices:
        import pygame.midi
        pygame.midi.init()
        for i in range(pygame.midi.get_count()):
            interf, name, is_input, is_output, opened = pygame.midi.get_device_info(i)
            kind = "output" if is_output else "input"
            print(f"[{i}] {name.decode()} ({kind})")
        pygame.midi.quit()
        return

    if not os.path.exists(args.hand_model):
        raise SystemExit(
            f"Hand landmark model not found at {args.hand_model}\n"
            f"Download it with:\n"
            f"  curl -L -o {args.hand_model} "
            f"https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            f"hand_landmarker/float16/1/hand_landmarker.task"
        )

    model = tf.keras.models.load_model(args.model)
    with open(args.labels) as f:
        class_labels = json.load(f)
    label_map = build_label_map(class_labels)
    print("Gesture index mapping in use:", label_map)

    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.hand_model),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)  # fallback for non-macOS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    left_stabilizer = GestureStabilizer(window=args.smoothing_window)
    right_stabilizer = GestureStabilizer(window=args.smoothing_window)

    engine = MusicEngine(instrument_program=0, device_id=args.midi_device)  # 0 = Acoustic Grand Piano

    prev_time = time.time()
    fps = 0.0

    print("Running. Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            # Flip for a natural selfie view; MediaPipe handedness is
            # computed on this flipped frame so it matches the user's
            # actual left/right hands as they see themselves.
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_h, frame_w = frame.shape[:2]
            left_gesture_raw, left_conf = None, 0.0
            right_gesture_raw, right_conf = None, 0.0

            if result.hand_landmarks and result.handedness:
                for hand_landmarks, handedness in zip(
                        result.hand_landmarks, result.handedness):
                    hand_label = handedness[0].category_name  # "Left" / "Right"

                    x1, y1, x2, y2 = landmarks_bbox(hand_landmarks, frame_w, frame_h)
                    crop = frame[y1:y2, x1:x2]
                    gesture_idx, conf = classify_hand_crop(model, class_labels, label_map, crop)

                    draw_landmarks(frame, hand_landmarks)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)

                    if hand_label == "Left":
                        left_gesture_raw, left_conf = gesture_idx, conf
                    else:
                        right_gesture_raw, right_conf = gesture_idx, conf

            left_stable, _ = left_stabilizer.update(left_gesture_raw, left_conf)
            right_stable, _ = right_stabilizer.update(right_gesture_raw, right_conf)

            both_present = left_stable is not None and right_stable is not None
            if both_present:
                engine.set_state(left_stable, right_stable)
                status = "Playing"
            else:
                engine.set_state(None, None)
                status = "Waiting for both hands"

            chord_name, style_name = engine.current_labels()
            engine.tick()

            left_label = (GESTURE_TO_CHORD[left_stable][0]
                          if left_stable in GESTURE_TO_CHORD else "-")
            right_label = (GESTURE_TO_STYLE[right_stable][0]
                            if right_stable in GESTURE_TO_STYLE else "-")
            left_info = f"{left_label} ({left_conf:.2f})"
            right_info = f"{right_label} ({right_conf:.2f})"

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            frame = draw_ui(frame, fps, left_info, right_info,
                             chord_name, style_name, status)
            cv2.imshow("Two-Hand Gesture Instrument", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        engine.shutdown()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()