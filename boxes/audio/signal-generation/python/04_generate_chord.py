# Generate a chord (a combination of tones) 
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

# Generate chord from a list of notes
duration = 2.0
c_major = ['C4', 'E4', 'G4']
e_minor = ['E4', 'G4', 'B4']
c_major7 = ['C4', 'E4', 'G4', 'B4']
names = c_major
notes = []
for name in names:
    freq = Utilities.NOTE_FREQS[name]
    note = Utilities.generate_note(duration, freq, sample_rate, num_channels=num_channels, num_harmonics=5)
    notes.append(note)
notes = np.array(notes)
chord = np.sum(notes, axis=0)
chord = chord / np.max(np.abs(chord)) # Avoid clipping

# Wait to save recording
input("Press <Enter> to start generation...")

# Send sound to speaker
speaker.write(chord)

# Wait for sound output to finish
try:
    while speaker.is_playing():
        time.sleep(0.01)
finally:
    speaker.stop()

# ---------------------------------
# Plot spectrum (using FFT) of tone
# ---------------------------------

# Compute frequency spectrum
freqs, mags = Utilities.compute_spectrum(chord, sample_rate, min_freq=10, max_freq=10000)

# Find peaks
peaks_x, peaks_y = Utilities.find_peaks(freqs, mags)

# Plot frequency spectrum
plt.figure()
plt.tight_layout()
plt.plot(freqs, mags)
for px, py in zip(peaks_x, peaks_y):
    plt.plot(px, py, 'ro')
    plt.text(px + 150.0, py, f"{px:.1f} Hz", ha='left', va='center', fontsize=8)
plt.title(f"Chord Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)

# Save frequency spectrum plot
save_path = f"{project_path}/my_chord_frequency_spectrum.png"
plt.savefig(f"{save_path}")

#FIN