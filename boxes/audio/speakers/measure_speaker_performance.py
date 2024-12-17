# Measure Speaker Performance
# - Frequency Response
# - Distortion

# Import libraries
import os
import time
import importlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# Import modules
import NB3.Sound.utilities as Utilities
import NB3.Sound.speaker as Speaker
import NB3.Sound.microphone as Microphone

# Reload modules
result = importlib.reload(Microphone)
result = importlib.reload(Speaker)
result = importlib.reload(Utilities)

# Get user name
username = os.getlogin()

# Specify the output folder
output_folder = f"/home/{username}/NoBlackBoxes/LastBlackBox/_tmp/sounds/speakers"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List available sound devices
Utilities.list_devices()

# Get output device
output_device = Utilities.get_output_device_by_name("default")

# Get input device
input_device = Utilities.get_input_device_by_name("Blue Snowball")

# Specify speaker params
speaker_num_channels = 2
speaker_sample_rate = 48000
speaker_buffer_size = int(speaker_sample_rate / 10)
speaker_max_samples = int(speaker_sample_rate * 10)

# Specify microphone params
microphone_num_channels = 1
microphone_sample_rate = 48000
microphone_buffer_size = int(microphone_sample_rate / 10)
microphone_max_samples = int(microphone_sample_rate * 10)

# Generate sine wave sweep (20 Hz to 20,000 Hz)
duration = 10.0
sample_rate = 48000
start_frequency = 20.0
stop_frequency = 20000.0
sweep = Utilities.generate_frequency_sweep(duration, start_frequency, stop_frequency, speaker_sample_rate, speaker_num_channels)

# Initialize speaker
speaker = Speaker.Speaker(output_device, speaker_num_channels, 'int32', speaker_sample_rate, speaker_buffer_size)
speaker.start()

# Initialize microphone
microphone = Microphone.Microphone(input_device, microphone_num_channels, 'int32', microphone_sample_rate, microphone_buffer_size, microphone_max_samples)
microphone.start()

# Send sound speaker
speaker.write(sweep)

# Wait for output to finish
while speaker.is_playing():
    time.sleep(0.1)
sweep_recording = microphone.latest(microphone.valid_samples)

# Store recording
microphone.save_wav(f"{output_folder}/sweep_recording.wav", microphone.valid_samples)

# Generate pure tones
duration = 3.0
sample_rate = 48000
frequency = 440.0
tone = Utilities.generate_pure_tone(duration, frequency, speaker_sample_rate, speaker_num_channels)

# Reset recording
microphone.reset()

# Send sound speaker
speaker.write(tone)

# Wait for output to finish
while speaker.is_playing():
    time.sleep(0.1)
tone_recording = microphone.latest(microphone.valid_samples)

# Shutdown
speaker.stop()
microphone.stop()

recording = tone_recording

# Process Recording
normalized = (recording / np.max(np.abs(recording)))[:,0]
frequencies, times, Sxx = signal.spectrogram(normalized, fs=speaker_sample_rate, nperseg=1024, nfft=2048)
#frequencies, times, Sxx = signal.spectrogram(sweep[:,0], fs=speaker_sample_rate, nperseg=1024, nfft=2048)

# Convert power spectrogram to dB scale
Sxx_dB = 10 * np.log10(Sxx + 1e-10)  # Small epsilon to avoid log(0)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies, Sxx_dB, shading='gouraud')
plt.colorbar(label="Power (dB)")
plt.title("Spectrogram of Recorded Linear Chirp")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.ylim(0, speaker_sample_rate / 2)  # Limit y-axis to Nyquist frequency
plt.tight_layout()
plt.savefig(f"{output_folder}/sweep_spectrogram.png", dpi=300, bbox_inches="tight")

# Average power spectral density across time
average_power = np.mean(Sxx, axis=1)  # Collapse time axis to get average at each frequency

# Convert to dB scale
average_power_dB = 10 * np.log10(average_power + 1e-10)  # Avoid log(0)

# Plot frequency response curve
plt.figure(figsize=(10, 6))
plt.semilogx(frequencies, average_power_dB, label="Frequency Response Curve", color="b", linewidth=2)
plt.title("Frequency Response Curve (Log Scale)")
plt.xlabel("Frequency [Hz] (Log Scale)")
plt.ylabel("Power (dB)")
plt.grid(which="both", linestyle="--", linewidth=0.5)  # Grid for both major/minor ticks
plt.xlim(20, speaker_sample_rate / 2)  # Limit x-axis to relevant range
plt.tight_layout()
plt.legend()
plt.savefig(f"{output_folder}/sweep_response.png", dpi=300, bbox_inches="tight")

# FIN
