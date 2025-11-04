import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Locals libs
import NB3.Sound.microphone as Microphone
import NB3.Sound.utilities as Utilities

# Reimport
import importlib
importlib.reload(Microphone)
importlib.reload(Utilities)

# Get user name
username = os.getlogin()

# Specify paths
repo_path = f"/home/{username}/NoBlackBoxes/LastBlackBox"
project_path = f"{repo_path}/boxes/audio/signal-processing/python"

# Specify params
input_device = 1
num_channels = 2
sample_rate = 48000
buffer_size = int(sample_rate / 10)
max_samples = int(sample_rate * 5)

# List available sound devices
Utilities.list_devices()

# Initialize microphone
microphone = Microphone.Microphone(input_device, num_channels, 'int32', sample_rate, buffer_size, max_samples)
microphone.gain = 10.0
microphone.start()

# Clear error ALSA/JACK messages from terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Wait to save recording
input("Press <Enter> to start 5 second recording...")

# Live processing
for i in range(50): # 50 buffers (10 per second)
    latest = microphone.latest(buffer_size)
    if num_channels == 2:
        left_volume = np.mean(np.abs(latest[:,0]))
        right_volume = np.mean(np.abs(latest[:,1]))
        print("{0:.2f} {1:.2f}".format(left_volume, right_volume))
    else:
        volume = np.mean(np.abs(latest[:]))
        print("{0:.2f}".format(volume))
    time.sleep(0.1)

# Store recording
recording = np.copy(microphone.sound)

# Compute FFT
n = len(recording)
freqs = np.fft.fftfreq(n, d=1/sample_rate)[:n // 2]
fft_left = np.fft.fft(recording[:,0])           # FFT left channel
fft_left = np.abs(fft_left[:n // 2])            # Take the positive frequencies
fft_right = np.fft.fft(recording[:,1])          # FFT right channel
fft_right = np.abs(fft_right[:n // 2])          # Take the positive frequencies

# Normalize FFT
fft_left = fft_left / np.max(fft_left)
fft_right = fft_right / np.max(fft_right)

# Shutdown microphone
microphone.stop()

# Plot frequency spectrum
plt.plot(freqs, fft_left)
plt.plot(freqs, fft_right)
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Amplitude")
plt.xlim(10, 10_000)   # <-- Limit plot to 0–10 kHz
plt.grid(True)

# Save frequency spectrum
save_path = f"{project_path}/my_frequency_measurement.png"
plt.savefig(f"{save_path}")

#FIN