"""
gesture_utils.py

Shared utilities for the two-hand gesture musical instrument:
  - Landmark -> feature vector conversion (used identically for
    training and for real-time inference, so the model never sees
    a distribution mismatch).
  - Temporal smoothing / debouncing so a single noisy frame can't
    change the chord or performance style.
"""

from collections import deque, Counter
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Hand skeleton connections (for manual drawing) — 21-point MediaPipe
# hand topology. Defined locally since the Tasks API (unlike the old
# `mp.solutions` API) doesn't ship a drawing helper.
# ---------------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def landmarks_to_features(hand_landmarks) -> np.ndarray:
    """
    Convert a list of 21 landmark points (each with .x/.y/.z, as
    returned per-hand by MediaPipe's Tasks API) into a translation-
    and scale-invariant feature vector.

    Invariance matters because the same gesture should classify the
    same way regardless of where the hand is in the frame or how
    close it is to the camera.

    Returns a (63,) float32 array: 21 landmarks x (x, y, z), with the
    wrist (landmark 0) subtracted out and the whole hand scaled by
    the distance from wrist to middle-finger MCP (landmark 9).
    """
    pts = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32,
    )

    wrist = pts[0].copy()
    pts -= wrist  # translation invariance

    scale = np.linalg.norm(pts[9])  # wrist -> middle finger MCP
    if scale < 1e-6:
        scale = 1e-6
    pts /= scale  # scale invariance

    return pts.flatten()  # shape (63,)


def landmarks_to_features_from_array(raw_xyz: np.ndarray) -> np.ndarray:
    """
    Same normalization as `landmarks_to_features`, but for a raw
    (21, 3) numpy array instead of a MediaPipe object. Used by the
    training script when re-deriving landmarks isn't needed live.
    """
    pts = raw_xyz.astype(np.float32).copy()
    wrist = pts[0].copy()
    pts -= wrist
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = 1e-6
    pts /= scale
    return pts.flatten()


# ---------------------------------------------------------------------------
# CNN preprocessing (shared by train_gesture_cnn.py and main.py so training
# and live inference see IDENTICAL preprocessing — critical, since any
# mismatch here silently tanks live accuracy even if training looked fine).
# ---------------------------------------------------------------------------
CNN_IMAGE_SIZE = 64  # width/height fed into the CNN


def preprocess_for_cnn(gray_img: np.ndarray, target_size: int = CNN_IMAGE_SIZE) -> np.ndarray:
    """
    gray_img: single-channel (H, W) uint8 image, already a binarized /
    thresholded silhouette (0 or 255). Resizes and normalizes to the
    (target_size, target_size, 1) float32 array the CNN expects.
    """
    resized = cv2.resize(gray_img, (target_size, target_size),
                          interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    return arr.reshape(target_size, target_size, 1)


def binarize_hand_crop(bgr_crop: np.ndarray) -> np.ndarray:
    """
    Converts a color webcam crop of a hand into a black/white silhouette
    approximating the training dataset's binarized style, using Otsu's
    threshold on grayscale. If your dataset was binarized differently
    (e.g. background-subtracted, different polarity), adjust this to
    match — the closer this matches your training data, the better
    live accuracy will be.
    """
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def landmarks_bbox(hand_landmarks, frame_w: int, frame_h: int, padding: float = 0.3):
    """
    Returns a pixel-space (x1, y1, x2, y2) bounding box around a hand's
    landmarks, padded and clamped to the frame, so we can crop the hand
    region out of the live webcam frame for the CNN.
    """
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    box_w = x_max - x_min
    box_h = y_max - y_min
    x_min -= box_w * padding
    x_max += box_w * padding
    y_min -= box_h * padding
    y_max += box_h * padding

    x1 = max(0, int(x_min * frame_w))
    y1 = max(0, int(y_min * frame_h))
    x2 = min(frame_w, int(x_max * frame_w))
    y2 = min(frame_h, int(y_max * frame_h))
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------
class GestureStabilizer:
    """
    Debounces raw per-frame gesture predictions.

    A new gesture is only reported as "active" once it wins a
    majority vote over the last `window` frames AND its average
    confidence over those frames clears `min_confidence`. This
    prevents flicker-triggered chord/style changes and repeated
    note triggering from unstable detections.
    """

    def __init__(self, window: int = 8, min_confidence: float = 0.6,
                 min_agreement: float = 0.6):
        self.window = window
        self.min_confidence = min_confidence
        self.min_agreement = min_agreement
        self._preds = deque(maxlen=window)
        self._confs = deque(maxlen=window)
        self.stable_gesture = None  # last accepted stable gesture (or None)

    def update(self, gesture_id, confidence: float):
        """
        Feed one new (possibly None) frame prediction in.
        Returns (stable_gesture, changed: bool).
        `changed` is True only on the frame where the stable gesture
        actually flips to a new value.
        """
        self._preds.append(gesture_id)
        self._confs.append(confidence)

        if len(self._preds) < self.window:
            return self.stable_gesture, False

        counts = Counter(g for g in self._preds if g is not None)
        if not counts:
            changed = self.stable_gesture is not None
            self.stable_gesture = None
            return None, changed

        winner, win_count = counts.most_common(1)[0]
        agreement = win_count / self.window
        avg_conf = float(np.mean([c for g, c in zip(self._preds, self._confs)
                                   if g == winner]))

        if agreement >= self.min_agreement and avg_conf >= self.min_confidence:
            changed = winner != self.stable_gesture
            self.stable_gesture = winner
            return winner, changed

        # Not confident/agreed enough to switch -> keep previous stable value
        return self.stable_gesture, False

    def reset(self):
        self._preds.clear()
        self._confs.clear()
        self.stable_gesture = None
