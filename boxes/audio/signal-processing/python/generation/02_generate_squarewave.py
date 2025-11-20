# Generate a square wave
import os, time
import matplotlib.pyplot as plt
import LBB.config as Config
import NB3.Sound.speaker as Speaker
import NB3.Sound.utilities as Utilities

# Specify paths
project_path = f"{Config.repo_path}/boxes/audio/signal-processing/python/generation"

# List available sound devices
Utilities.list_devices()

# Get speaker device by name (NB3: "MAX", PC: select based on listed output devices)
output_device = Utilities.get_output_device_by_name("MAX")
if output_device == -1:
    exit("Output device not found")

# Specify speaker params
num_channels = 2
sample_rate = 48000
buffer_size = int(sample_rate / 10)

# Initialize speaker
speaker = Speaker.Speaker(output_device, num_channels, 'int32', sample_rate, buffer_size)
speaker.start()

# Clear error ALSA/JACK messages from terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Generate square wave
duration = 2.0
frequency = 440.0
square = Utilities.generate_square_wave(duration, frequency, sample_rate, num_channels, duty_cycle=0.5)

# Wait to save recording
input("Press <Enter> to start generation...")

# Send sound to speaker
speaker.write(square)

# Wait for sound output to finish
try:
    while speaker.is_playing():
        time.sleep(0.01)
finally:
    speaker.stop()

# ----------------------------------------
# Plot spectrum (using FFT) of square wave
# ----------------------------------------

# Compute frequency spectrum
freqs, mags = Utilities.compute_spectrum(square, sample_rate, min_freq=10, max_freq=10000)

# Find peaks
peaks_x, peaks_y = Utilities.find_peaks(freqs, mags, min_prominence=0.05)

# Plot frequency spectrum
plt.figure()
plt.tight_layout()
plt.plot(freqs, mags)
for px, py in zip(peaks_x, peaks_y):
    plt.plot(px, py, 'ro')
    plt.text(px + 150.0, py, f"{px:.1f} Hz", ha='left', va='center', fontsize=6)
plt.title("Square Wave Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)

# Save frequency spectrum plot
save_path = f"{project_path}/my_square_wave_frequency_spectrum.png"
plt.savefig(f"{save_path}")

#FIN