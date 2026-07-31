# 🎵 Two-Hand Gesture Musical Instrument

Play music with your webcam. **Left hand picks the chord, right hand picks how it's played** — no MIDI controller, no keyboard, no setup beyond a webcam and Python.

```
Left hand  →  Chord        (C Major, D Minor, G7, ...)
Right hand →  Style        (Full Chord, Arpeggio, Strum, Tremolo, ...)
```

Audio is synthesized live with `pygame.mixer` — no MIDI drivers, no FluidSynth, no virtual ports. It just plays through your speakers.

---

## How it works

```
Webcam
  │
  ▼
MediaPipe HandLandmarker  ──►  locates both hands, tells left vs. right
  │
  ▼
Crop each hand's bounding box  ──►  binarize to black/white silhouette
  │
  ▼
CNN classifier (gesture_cnn_model.keras)  ──►  gesture id + confidence
  │
  ▼
GestureStabilizer (per hand)  ──►  majority-vote smoothing over N frames
  │
  ▼
MusicEngine.set_state(chord, style)  ──►  synthesizes + plays audio
```

Music only plays once **both hands** are detected with a stable, confident gesture. Press `q` to quit.

---

## Project structure

```
.
├── main.py                        # Webcam loop — ties detection, classification, and audio together
├── music_engine.py                # Chord/style tables + real-time audio synthesis (pygame.mixer)
├── gesture_utils.py                # Hand-crop preprocessing + temporal smoothing (shared by training & live app)
├── train_gesture_cnn.py           # Trains the CNN on binarized dataset images (the trainer actually used)
├── train_gesture_classifier.py    # Landmark-based alternative trainer — kept for reference only
├── gesture_cnn_model.keras        # Trained CNN weights
├── gesture_cnn_labels.json        # Class label order matching the trained model
├── hand_landmarker.task            # MediaPipe hand-detection model
├── requirements.txt
├── dataset/                       # Training images (not tracked in git — see below)
│   └── train/
│       └── train/
│           ├── 0/
│           ├── 1/
│           ├── ...
│           └── 19/
└── tests/                         # Small standalone scripts for debugging camera/audio in isolation
    ├── cv_only_test.py
    ├── engine_only_test.py
    ├── gesture_change_test.py
    ├── camera_plus_engine_test.py
    └── raw_compare_test.py
```

> **About the `dataset/` folder:** each subfolder name (`0`–`19`) is a gesture class, containing binarized (black/white silhouette) training images. This layout is what `train_gesture_cnn.py` expects out of the box. The dataset isn't included in this repo — see [`.gitignore`](#gitignore-recommendation) below.

---

## 1. Install

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

Requires Python 3.10+ (tested with TensorFlow 2.15+). No MIDI setup, no FluidSynth, no IAC Driver — audio works out of the box on macOS, Windows, and Linux.

You'll also need the MediaPipe hand landmark model:

```bash
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

---

## 2. Train the gesture classifier

Your dataset is binarized (black/white silhouette images), which MediaPipe's hand detector can't reliably locate a hand in. `train_gesture_cnn.py` works directly on the binarized pixels instead:

```bash
python3 train_gesture_cnn.py --dataset dataset/train/train --output gesture_cnn_model.keras
```

This saves `gesture_cnn_model.keras` plus `gesture_cnn_labels.json`, and prints the exact class names found:

```
Classes (20): ['0', '1', '10', '11', ...]
```

> `train_gesture_classifier.py` (the landmark-based trainer) is kept for reference — it depends on MediaPipe finding a hand in the image, which fails almost universally on silhouettes.

---

## 3. Map gestures to chords/styles

Open `main.py` and fill in `GESTURE_LABEL_MAP` near the top, using the class names printed in step 2, assigning each to an index 1–10:

```python
GESTURE_LABEL_MAP = {
    "0": 1,   # -> C Major (left hand) / Full Chord (right hand)
    "1": 2,   # -> D Minor / Ascending Arpeggio
    "10": 3,  # -> E Minor / Descending Arpeggio
    ...
}
```

The dataset has **20** classes, but `GESTURE_TO_CHORD` / `GESTURE_TO_STYLE` only define **10** slots. Pick the 10 clearest/most reliable classes and leave the rest out of `GESTURE_LABEL_MAP` so they're ignored.

The same 10 gestures are reused for both hands — which hand performs the gesture (detected automatically) decides whether it picks a **chord** (`GESTURE_TO_CHORD` in `music_engine.py`) or a **performance style** (`GESTURE_TO_STYLE`). Edit those two dicts directly for custom chords/styles.

If you skip this step, the app still runs — gestures auto-wrap into indices 1–10 in alphabetical order. Fine for a smoke test, not for a real mapping.

---

## 4. Run it

```bash
python3 main.py --model gesture_cnn_model.keras --labels gesture_cnn_labels.json
```

- Show your **left hand** to pick a chord, your **right hand** to pick how it's performed.
- Music only plays once **both hands** are detected with a stable, confident gesture.
- Press `q` to quit.

### Useful flags

| Flag | Purpose |
|---|---|
| `--camera` | Camera index (default `0`) |
| `--smoothing-window` | Frames of majority-vote smoothing per hand (default `8`) — lower = snappier/twitchier, higher = steadier/slower |
| `--hand-model` | Path to `hand_landmarker.task` |
| `--width` / `--height` | Capture resolution |

Other tuning knobs live directly in `main.py`:
- `CONFIDENCE_THRESHOLD` — raise it if wrong chords keep triggering, lower it if valid gestures get rejected too often.
- `min_detection_confidence` / `min_tracking_confidence` — MediaPipe's own hand-detection strictness.

---

## Files reference

| File | Purpose |
|---|---|
| `main.py` | Webcam loop, hand detection/crop, CNN classification, UI, ties everything together |
| `music_engine.py` | Chord/style definitions, direct audio synthesis (`pygame.mixer`), real-time performance patterns |
| `gesture_utils.py` | Hand-crop preprocessing + temporal smoothing, shared by training and the live app |
| `train_gesture_cnn.py` | Binarized dataset images → trained CNN (`gesture_cnn_model.keras` + `gesture_cnn_labels.json`) |
| `train_gesture_classifier.py` | Landmark-based alternative trainer — reference only, doesn't work on binarized datasets |

---

## Troubleshooting

- **No sound / silent playback:** confirm `pygame.mixer.get_init()` succeeds and your system's default output device is correct — this project intentionally avoids MIDI/FluidSynth because that path was unreliable on macOS in testing (MIDI messages reached the synth, but audio was inconsistently silent). Direct synthesis via `pygame.mixer` sidesteps that.
- **Wrong / flickering chords:** raise `CONFIDENCE_THRESHOLD` or `--smoothing-window` in `main.py`.
- **Camera won't open on macOS:** the app defaults to `cv2.CAP_AVFOUNDATION`; try a different `--camera` index if you have multiple devices.
- **Import order matters:** `music_engine` (and `pygame`) must be imported *before* `cv2` — both bundle their own copy of SDL2 on macOS, and importing `cv2` first can silently break `pygame`'s audio internals. `main.py` already does this correctly; keep the order if you refactor.

---

## `.gitignore` recommendation

The dataset and trained model are large binary artifacts — track them with Git LFS or keep them out of version control entirely:

```gitignore
dataset/
*.keras
*.task
__pycache__/
*.pyc
.venv/
```

If you want the trained model available to collaborators without LFS, consider hosting `gesture_cnn_model.keras` and `hand_landmarker.task` as GitHub Release assets instead of committing them directly.

---

## Roadmap (not in this MVP)

Multiple instruments, drum mode, loop recording, gesture recording/playback, user-remappable gestures, tempo/volume control via gesture. The architecture (separate stabilizer per hand, `MusicEngine.set_state`, style-handler dict) is built so these can be layered on without a rewrite.

## License

Add your license of choice here (e.g. MIT).
