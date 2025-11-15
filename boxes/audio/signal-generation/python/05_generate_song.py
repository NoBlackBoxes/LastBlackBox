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

# Initialize speaker
speaker = Speaker.Speaker(output_device, num_channels, 'int32', sample_rate, buffer_size)
speaker.start()

# Clear error ALSA/JACK messages from terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Specify standard note durations
tempo = 60          # Beats per minute (Bach)
#tempo = 120         # Beats per minute (Axel)
beat = 60.0/tempo   # Beat duration (seconds)
sb = 4.0*beat       # Whole note (semibreve)
mi = 2.0*beat       # Half note (minim)
dc = 1.5*beat       # Dotted quarter note (dotted crotchet)
cr = beat           # Quarter note (crotchet)
dq = (3.0*beat)/4.0 # Dotted 8th note (dotted quaver)
qu = beat/2.0       # 8th note (quaver)
sq = beat/4.0       # 16th note (semi-quaver)

# List song notes and durations
bach_prelude_c_major_intro = [
    # Bar 1/2 (C major)
    ('C4',sq), ('E4',sq), ('G4',sq), ('C5',sq),
    ('E5',sq), ('G4',sq), ('C5',sq), ('E5',sq),
    ('C4',sq), ('E4',sq), ('G4',sq), ('C5',sq),
    ('E5',sq), ('G4',sq), ('C5',sq), ('E5',sq),

    # Bar 3/4 (D minor)
    ('D4',sq), ('F4',sq), ('A4',sq), ('D5',sq),
    ('F5',sq), ('A4',sq), ('D5',sq), ('F5',sq),
    ('D4',sq), ('F4',sq), ('A4',sq), ('D5',sq),
    ('F5',sq), ('A4',sq), ('D5',sq), ('F5',sq),

    # Bar 5/6 (G major7 flavor)
    ('G3',sq), ('B3',sq), ('D4',sq), ('G4',sq),
    ('B4',sq), ('D4',sq), ('G4',sq), ('B4',sq),
    ('G3',sq), ('B3',sq), ('D4',sq), ('G4',sq),
    ('B4',sq), ('D4',sq), ('G4',sq), ('B4',sq),

    # Bar 7/8 (C major)
    ('C4',sq), ('E4',sq), ('G4',sq), ('C5',sq),
    ('E5',sq), ('G4',sq), ('C5',sq), ('E5',sq),
    ('C4',sq), ('E4',sq), ('G4',sq), ('C5',sq),
    ('E5',sq), ('G4',sq), ('C5',sq), ('E5',sq),
]
axel_f_intro = [
    ('D4', cr), ('F4', dq), ('D4', qu), ('D4', sq), ('G4', qu),  ('D4', qu), ('C4', qu),
    ('D4', cr), ('A4', dq), ('D4', qu), ('D4', sq), ('Bb4', qu), ('A4', qu), ('F4', qu), 
    ('D4', qu), ('A4', qu), ('D5', qu), ('D4', sq), ('C4', qu),  ('C4', sq), ('A3', qu), 
    ('E4', qu), ('D4', mi)
]

# Build song
song_notes = bach_prelude_c_major_intro
#song_notes = axel_f_intro
parts = []
for name, duration in song_notes:
    freq = Utilities.NOTE_FREQS[name]
    #note = Utilities.generate_pure_tone(duration, freq, sample_rate, num_channels=num_channels)
    #note = Utilities.generate_square_wave(duration, freq, sample_rate, num_channels=num_channels)
    note = Utilities.generate_note(duration, freq, sample_rate, num_channels=num_channels, num_harmonics=5)
    parts.append(note)
song = np.vstack(parts)
print(f"Song Duration: {song.shape[0]/sample_rate} seconds")

# Wait to save recording
input("Press <Enter> to start generation...")

# Send sound to speaker
speaker.write(song)

# Wait for sound output to finish
try:
    while speaker.is_playing():
        time.sleep(0.01)
except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    speaker.stop()

# ------------------------
# Plot spectrogram of song
# ------------------------

# Compute spectrogram
times, frequencies, magnitudes_db = Utilities.compute_spectrogram(song, sample_rate)

# Plot
plt.figure(figsize=(8, 4))
plt.pcolormesh(times, frequencies, magnitudes_db, shading='gouraud', cmap='plasma')
plt.ylim(10, 10000) # Limit plot range to 10 Hz → 10 kHz
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram (10 Hz to 10 kHz)')
plt.colorbar(label='Magnitude [dB]')
plt.tight_layout()

# Save frequency spectrum plot
save_path = f"{project_path}/my_song_spectrogram_measurement.png"
plt.savefig(f"{save_path}")

#FIN