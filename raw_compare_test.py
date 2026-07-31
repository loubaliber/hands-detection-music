import pygame.midi, time, sys

device_id = int(sys.argv[1])
pygame.midi.init()
out = pygame.midi.Output(device_id)

print("Test A: note_on with NO set_instrument call first...")
out.note_on(60, 100)
time.sleep(1.5)
out.note_off(60, 0)
time.sleep(0.5)

print("Test B: set_instrument(0) THEN note_on...")
out.set_instrument(0)
out.note_on(64, 100)
time.sleep(1.5)
out.note_off(64, 0)

out.close()
pygame.midi.quit()
print("Done.")
