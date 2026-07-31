import time
import sys
from music_engine import MusicEngine

device_id = int(sys.argv[1])
engine = MusicEngine(instrument_program=0, device_id=device_id)

print("Setting chord=1 (C Major), style=1 (Full Chord)...")
engine.set_state(1, 1)

start = time.time()
while time.time() - start < 6:
    engine.tick()
    time.sleep(0.03)  # mimic ~30fps loop timing

engine.shutdown()
print("Done.")
