# Generate a note (a fundamental tone + harmonics) 
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

# Generate a note (a fundamental frequency with harmonics falling off in amplitude)
duration = 2.0
C4 = 261.63
D4 = 293.66
E4 = 329.63
G4 = 392.0
A4 = 440.0
fundamental = D4
note = Utilities.generate_note(duration, fundamental, sample_rate, num_channels=2, num_harmonics=10)

# Wait to save recording
input("Press <Enter> to start generation...")

# Send sound to speaker
speaker.write(note)

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
freqs, mags = Utilities.compute_spectrum(note, sample_rate, min_freq=10, max_freq=10000)

# Find peaks
peaks_x, peaks_y = Utilities.find_peaks(freqs, mags)

# Plot frequency spectrum
plt.figure()
plt.tight_layout()
plt.plot(freqs, mags)
for px, py in zip(peaks_x, peaks_y):
    plt.plot(px, py, 'ro')
    plt.text(px + 150.0, py, f"{px:.1f} Hz", ha='left', va='center', fontsize=8)
plt.title(f"Note ({fundamental} Hz) with Harmonics Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)

# Save frequency spectrum plot
save_path = f"{project_path}/my_note_frequency_spectrum.png"
plt.savefig(f"{save_path}")

#FIN