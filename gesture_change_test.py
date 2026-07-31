import time, sys
from music_engine import MusicEngine

device_id = int(sys.argv[1])
engine = MusicEngine(instrument_program=0, device_id=device_id)

print("State 1: C Major, Full Chord (holding 4s)...")
engine.set_state(1, 1)
start = time.time()
while time.time() - start < 4:
    engine.tick()
    time.sleep(0.03)

print("Switching to State 2: F Major, Ascending Arpeggio (holding 4s)...")
engine.set_state(4, 2)
start = time.time()
while time.time() - start < 4:
    engine.tick()
    time.sleep(0.03)

print("Switching BACK to State 1: C Major, Full Chord (holding 4s)...")
engine.set_state(1, 1)
start = time.time()
while time.time() - start < 4:
    engine.tick()
    time.sleep(0.03)

engine.shutdown()
print("Done.")
