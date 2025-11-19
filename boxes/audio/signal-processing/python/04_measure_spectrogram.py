# Measure and plot the spectrogram (frequency vs time) of a sound recorded by the microphone
import os, pathlib, time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import NB3.Sound.microphone as Microphone
import NB3.Sound.utilities as Utilities

# Specify paths
repo_path = f"{pathlib.Path.home()}/NoBlackBoxes/LastBlackBox"
project_path = f"{repo_path}/boxes/audio/signal-processing/python"

# List available sound devices
Utilities.list_devices()

# Get microphone device by name (NB3: "MAX", PC: select based on listed input devices)
input_device = Utilities.get_input_device_by_name("HD-Audio")
if input_device == -1:
    exit("Input device not found")

# Specify microphone params
num_channels = 2
sample_rate = 48000
buffer_size = int(sample_rate / 10)
max_samples = int(sample_rate * 5)

# Initialize microphone
microphone = Microphone.Microphone(input_device, num_channels, 'int32', sample_rate, buffer_size, max_samples)
microphone.gain = 10.0
microphone.start()

# Clear error ALSA/JACK messages from terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Wait to save recording
input("Press <Enter> to start 5 second recording...")

# Live volume processing
Utilities.meter_start()
for i in range(50):                                     # Process 50 buffers (10 per second)
    latest = microphone.latest(buffer_size)             # Get the latest audio buffer
    left_volume = np.mean(np.abs(latest[:,0]))          # Extract left channel volume (abs value of audio signal)
    right_volume = np.mean(np.abs(latest[:,1]))         # Extract right channel volume (abs value of audio signal)
    Utilities.meter_update(left_volume, right_volume)   # Update volume meter
    time.sleep(0.1) # Wait a bit
Utilities.meter_stop()

# Store recording
recording = np.copy(microphone.sound)

# Shutdown microphone
microphone.stop()

# Compute spectrogram using SciPy signal library
mono = np.mean(recording, axis=1)                   # Convert stereo to mono (average left and right channel)
frequencies, times, Sxx = sp.signal.spectrogram(
    mono,               # Audio signal (one channel)
    fs=sample_rate,     # Sample rate (Hz) of audio signal
    window='hann',      # Window applied to each audio segment before FFT (prevents edge effects)
    nperseg=1024,       # Number of samples per segment (measure FFT spectrum every on 1024 samples, best if power of 2)
    noverlap=512,       # Overlapping samples between each audio segment (the next segment overlaps a bit with the previous, improves time resolution)
    scaling='density',  # Output in power spectral density (PSD), units: V²/Hz
    mode='magnitude'    # Compute the magnitude (easiest to visualize)
)

# Convert to dB
Sxx_db = 20 * np.log10(Sxx + 1e-12)

# Plot
plt.figure(figsize=(8, 4))
plt.pcolormesh(times, frequencies, Sxx_db, shading='gouraud', cmap='plasma')
plt.ylim(20, 20000) # Limit plot range to 20 Hz → 20 kHz (typical human)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram (20 Hz to 20 kHz)')
plt.colorbar(label='Magnitude [dB]')
plt.tight_layout()

# Save frequency spectrum plot
save_path = f"{project_path}/my_spectrogram_measurement.png"
plt.savefig(f"{save_path}")

#FIN