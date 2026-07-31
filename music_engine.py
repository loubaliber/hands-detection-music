"""
music_engine.py

Turns a (chord, performance style) pair into real-time AUDIO output,
synthesized directly with pygame.mixer -- no MIDI, no FluidSynth, no
virtual MIDI ports involved.

Why: on this setup, MIDI messages reached FluidSynth correctly
(confirmed via voice_count > 0 during playback) but audio was
inconsistently inaudible regardless of routing/config -- a
CoreMIDI/CoreAudio-side issue outside the app's control. Synthesizing
tones directly and playing them through pygame.mixer removes that
entire dependency chain.

Design: `MusicEngine.set_state(...)` loads a repeating "step sequence"
for the current (chord, style) pair (same approach as before).
`MusicEngine.tick()` must be called every iteration of the main loop;
it checks elapsed time and fires the next note on/off step when due.
"""

import time

import numpy as np
import pygame

SAMPLE_RATE = 44100
MIXER_CHANNELS = 32  # simultaneous sounds available


# ---------------------------------------------------------------------------
# Music theory: chord definitions (root note name + interval pattern)
# ---------------------------------------------------------------------------
NOTE_TO_SEMITONE = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
    "B": 11,
}

CHORD_INTERVALS = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "maj7": [0, 4, 7, 11],
    "dom7": [0, 4, 7, 10],
}

GESTURE_TO_CHORD = {
    1: ("C Major", "C", "maj"),
    2: ("D Minor", "D", "min"),
    3: ("E Minor", "E", "min"),
    4: ("F Major", "F", "maj"),
    5: ("G Major", "G", "maj"),
    6: ("A Minor", "A", "min"),
    7: ("B Diminished", "B", "dim"),
    8: ("Cmaj7", "C", "maj7"),
    9: ("G7", "G", "dom7"),
    10: ("Fmaj7", "F", "maj7"),
}

# gesture index -> (style name, tempo in seconds-per-event)
GESTURE_TO_STYLE = {
    1: ("Full Chord", 0.0),
    2: ("Ascending Arpeggio", 0.18),
    3: ("Descending Arpeggio", 0.18),
    4: ("Broken Chord", 0.22),
    5: ("Bass Note", 0.0),
    6: ("Melody Pattern", 0.20),
    7: ("Piano Roll", 0.10),
    8: ("Strum", 0.015),
    9: ("Tremolo", 0.09),
    10: ("Sustain", 0.0),
}

BASE_OCTAVE = 4  # middle-ish register; MIDI C4 = 60


def chord_to_midi_notes(root: str, quality: str, octave: int = BASE_OCTAVE):
    root_semitone = NOTE_TO_SEMITONE[root]
    base = (octave + 1) * 12 + root_semitone
    return [base + i for i in CHORD_INTERVALS[quality]]


def midi_to_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def _generate_tone(freq: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Builds a short, seamlessly-loopable stereo waveform for `freq` using
    a few harmonics (piano-ish timbre). The buffer holds an EXACT
    integer number of periods so looping it (loops=-1) produces no
    click/discontinuity at the wrap point.
    """
    period_samples = sample_rate / freq
    target_len = int(sample_rate * 0.3)  # ~0.3s buffer
    n_periods = max(1, round(target_len / period_samples))
    total_samples = int(round(n_periods * period_samples))
    t = np.arange(total_samples) / sample_rate

    wave = (
        1.00 * np.sin(2 * np.pi * freq * t) +
        0.50 * np.sin(2 * np.pi * freq * 2 * t) +
        0.25 * np.sin(2 * np.pi * freq * 3 * t) +
        0.125 * np.sin(2 * np.pi * freq * 4 * t)
    )
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak
    wave = (wave * 0.35 * 32767).astype(np.int16)
    stereo = np.column_stack([wave, wave])
    return np.ascontiguousarray(stereo)


# ---------------------------------------------------------------------------
# Step sequence builders. Each returns a list of
# (wait_seconds, notes_to_turn_on, notes_to_turn_off) tuples forming
# ONE repeating cycle. `wait_seconds` is how long to hold AFTER firing
# that step before the next one fires. MusicEngine loops the list
# indefinitely until set_state() loads a new one.
# ---------------------------------------------------------------------------
def _build_sequence_pattern(seq, step):
    n = len(seq)
    steps = []
    for i in range(n):
        prev = seq[i - 1]  # wraps to seq[-1] when i == 0
        steps.append((step, [seq[i]], [prev]))
    return steps


def build_steps(style_name: str, notes, step: float):
    if style_name == "Sustain":
        return [(999999.0, notes, [])]

    if style_name == "Full Chord":
        return [(0.9, notes, []), (0.05, [], notes)]

    if style_name == "Bass Note":
        root = notes[0] - 12
        return [(1.0, [root], []), (0.05, [], [root])]

    if style_name == "Ascending Arpeggio":
        return _build_sequence_pattern(list(notes), step)

    if style_name == "Descending Arpeggio":
        return _build_sequence_pattern(list(reversed(notes)), step)

    if style_name == "Piano Roll":
        return _build_sequence_pattern(list(notes), step)

    if style_name == "Broken Chord":
        if len(notes) >= 3:
            pattern_idx = [0, 2, 1, 2]  # root, fifth, third, fifth
        elif len(notes) == 2:
            pattern_idx = [0, 1, 0, 1]
        else:
            pattern_idx = [0]
        seq = [notes[i % len(notes)] for i in pattern_idx]
        return _build_sequence_pattern(seq, step)

    if style_name == "Melody Pattern":
        idxs = [0, 1 % len(notes), 2 % len(notes), 1 % len(notes)]
        seq = [notes[i] + 12 for i in idxs]
        return _build_sequence_pattern(seq, step)

    if style_name == "Tremolo":
        root = notes[0]
        half = max(step / 2, 0.03)
        return [(half, [root], []), (half, [], [root])]

    if style_name == "Strum":
        steps = []
        for i, n in enumerate(notes):
            wait = step if i < len(notes) - 1 else 0.7
            steps.append((wait, [n], []))
        steps.append((0.05, [], notes))
        return steps

    return [(0.9, notes, []), (0.05, [], notes)]


class MusicEngine:
    def __init__(self, instrument_program: int = 0, device_id=None,
                 base_velocity: int = 90):
        # instrument_program / device_id kept as no-op params so existing
        # call sites (main.py) don't need to change; audio is synthesized
        # locally now, not routed through MIDI.
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
        pygame.mixer.set_num_channels(MIXER_CHANNELS)
        print("MusicEngine: using local audio synthesis (pygame.mixer), no MIDI")

        self.base_velocity = base_velocity
        self._sound_cache = {}  # note -> pygame.mixer.Sound
        self._note_channels = {}  # note -> pygame.mixer.Channel currently playing it

        self._chord_gesture = None
        self._style_gesture = None

        self._steps = []
        self._step_index = 0
        self._next_fire_time = None

    # -- public API ---------------------------------------------------
    def set_state(self, chord_gesture, style_gesture):
        if chord_gesture == self._chord_gesture and style_gesture == self._style_gesture:
            return  # no change

        self._chord_gesture = chord_gesture
        self._style_gesture = style_gesture
        self._silence_active_notes()
        self._steps = []
        self._step_index = 0
        self._next_fire_time = None

        if chord_gesture is None or style_gesture is None:
            return
        if chord_gesture not in GESTURE_TO_CHORD or style_gesture not in GESTURE_TO_STYLE:
            return

        _, root, quality = GESTURE_TO_CHORD[chord_gesture]
        style_name, step = GESTURE_TO_STYLE[style_gesture]
        notes = chord_to_midi_notes(root, quality)

        self._steps = build_steps(style_name, notes, step)
        if not self._steps:
            return

        wait, on_notes, off_notes = self._steps[0]
        for n in off_notes:
            self._note_off(n)
        for n in on_notes:
            self._note_on(n)
        self._next_fire_time = time.time() + wait

    def tick(self):
        """Call every iteration of the main loop."""
        if not self._steps or self._next_fire_time is None:
            return
        now = time.time()
        if now < self._next_fire_time:
            return

        self._step_index = (self._step_index + 1) % len(self._steps)
        wait, on_notes, off_notes = self._steps[self._step_index]
        for n in off_notes:
            self._note_off(n)
        for n in on_notes:
            self._note_on(n)
        self._next_fire_time = now + wait

    def current_labels(self):
        cg, sg = self._chord_gesture, self._style_gesture
        chord_name = GESTURE_TO_CHORD[cg][0] if cg in GESTURE_TO_CHORD else None
        style_name = GESTURE_TO_STYLE[sg][0] if sg in GESTURE_TO_STYLE else None
        return chord_name, style_name

    def shutdown(self):
        self._silence_active_notes()
        pygame.mixer.quit()

    # -- internals ------------------------------------------------------
    def _get_sound(self, note: int) -> pygame.mixer.Sound:
        sound = self._sound_cache.get(note)
        if sound is None:
            freq = midi_to_freq(note)
            arr = _generate_tone(freq)
            sound = pygame.sndarray.make_sound(arr)
            self._sound_cache[note] = sound
        return sound

    def _note_on(self, note: int, velocity: int = None):
        velocity = velocity if velocity is not None else self.base_velocity
        sound = self._get_sound(note)
        channel = pygame.mixer.find_channel(True)  # force=True: steal oldest if all busy
        channel.set_volume(min(velocity / 127.0, 1.0))
        channel.play(sound, loops=-1)  # loop indefinitely until we stop it
        self._note_channels[note] = channel

    def _note_off(self, note: int):
        channel = self._note_channels.pop(note, None)
        if channel is not None:
            channel.fadeout(80)  # ms; smooth release, avoids a click

    def _silence_active_notes(self):
        for note in list(self._note_channels.keys()):
            self._note_off(note)
