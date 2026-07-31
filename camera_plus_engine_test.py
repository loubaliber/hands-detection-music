import time, sys, cv2
from music_engine import MusicEngine

device_id = int(sys.argv[1])
camera_index = int(sys.argv[2]) if len(sys.argv) > 2 else 1

engine = MusicEngine(instrument_program=0, device_id=device_id)

print("Opening camera...")
cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
print("Camera opened:", cap.isOpened())

print("Setting chord=1, style=1 (Full Chord)...")
engine.set_state(1, 1)

start = time.time()
while time.time() - start < 8:
    ok, frame = cap.read()  # keep grabbing frames like the real app
    engine.tick()
    time.sleep(0.03)

engine.shutdown()
cap.release()
print("Done.")
