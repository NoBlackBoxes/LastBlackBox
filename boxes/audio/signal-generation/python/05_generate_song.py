# Generate a sequence of notes and chords...a song? 
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import NB3.Sound.speaker as Speaker
import NB3.Sound.utilities as Utilities

# Specify paths
username = os.getlogin()
repo_path = f"/home/{username}/NoBlackBoxes/LastBlackBox"
project_path = f"{repo_path}/boxes/audio/signal-generation/python"

# List available sound devices
Utilities.list_devices()

# Get output device
output_device = Utilities.get_output_device_by_name("HD-Audio Generic: ALC295 Analog")
#output_device = Utilities.get_output_device_by_name("MAX")
#output_device = Utilities.get_output_device_by_name("default")

# Specify speaker params
num_channels = 2
sample_rate = 48000
buffer_size = int(sample_rate / 10)
max_samples = int(sample_rate * 10)

# Initialize speaker
speaker = Speaker.Speaker(output_device, num_channels, 'int32', sample_rate, buffer_size)
speaker.start()

# Clear error ALSA/JACK messages from terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Generate song from a list of notes/chords
bach_prelude_c_major_intro = [
    # Bar 1 (C major)
    ('C4',0.25), ('E4',0.25), ('G4',0.25), ('C5',0.25),
    ('E5',0.25), ('G4',0.25), ('C5',0.25), ('E5',0.25),

    # Bar 2 (D minor)
    ('D4',0.25), ('F4',0.25), ('A4',0.25), ('D5',0.25),
    ('F5',0.25), ('A4',0.25), ('D5',0.25), ('F5',0.25),

    # Bar 3 (G major7 flavor)
    ('G3',0.25), ('B3',0.25), ('D4',0.25), ('G4',0.25),
    ('B4',0.25), ('D4',0.25), ('G4',0.25), ('B4',0.25),

    # Bar 4 (C major)
    ('C4',0.25), ('E4',0.25), ('G4',0.25), ('C5',0.25),
    ('E5',0.25), ('G4',0.25), ('C5',0.25), ('E5',0.25),
]
space = np.zeros((buffer_size, num_channels), dtype=np.float32)
parts = []
for name, duration in bach_prelude_c_major_intro:
    freq = Utilities.NOTE_FREQS[name]
    #note = Utilities.generate_note(duration, freq, sample_rate, num_channels=num_channels, num_harmonics=5)
    note = Utilities.generate_pure_tone(duration, freq, sample_rate, num_channels=num_channels)
    parts.append(note)
    parts.append(space)  # silence between notes
song = np.vstack(parts)

# Wait to save recording
input("Press <Enter> to start generation...")

# Send sound to speaker
speaker.write(song)

# Wait for sound output to finish
try:
    while speaker.is_playing():
        time.sleep(0.01)
finally:
    speaker.stop()

# ------------------------
# Plot spectrogram of song
# ------------------------

#FIN