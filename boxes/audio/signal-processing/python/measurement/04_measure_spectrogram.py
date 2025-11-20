# Measure and plot the spectrogram (frequency vs time) of a sound recorded by the microphone
import os, time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import LBB.config as Config
import NB3.Sound.microphone as Microphone
import NB3.Sound.utilities as Utilities

# Specify paths
project_path = f"{Config.repo_path}/boxes/audio/signal-processing/python/measurement"

# List available sound devices
Utilities.list_devices()

# Get microphone device by name (NB3: "MAX", PC: select based on listed input devices)
input_device = Utilities.get_input_device_by_name("MAX")
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

# Convert single channel (Mono)
if recording.ndim > 1:
    sound = np.mean(recording, axis=1)

# Compute spectrogram
times, frequencies, magnitudes_db = Utilities.compute_spectrogram(sound, sample_rate)

# Plot
plt.figure(figsize=(8, 4))
plt.pcolormesh(times, frequencies, magnitudes_db, shading='gouraud', cmap='plasma')
plt.ylim(10, 10000) # Limit plot range to 10 Hz → 10 kHz
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram (10 Hz to 10 kHz)')
plt.colorbar(label='Magnitude [dB]')
plt.tight_layout()

# Save spectrogram
save_path = f"{project_path}/my_spectrogram_measurement.png"
plt.savefig(f"{save_path}")

#FIN