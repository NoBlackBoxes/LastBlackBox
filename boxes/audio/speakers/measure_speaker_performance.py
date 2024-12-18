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
#output_device = Utilities.get_output_device_by_name("HD-Audio Generic: ALC295 Analog")
output_device = Utilities.get_output_device_by_name("MAX")
#output_device = Utilities.get_output_device_by_name("default")

# Get input device
#input_device = Utilities.get_input_device_by_name("HD-Audio Generic: ALC295 Analog")
input_device = Utilities.get_input_device_by_name("MAX")
#input_device = Utilities.get_input_device_by_name("Blue Snowball")

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
duration = 5.0
sample_rate = 48000
start_frequency = 20.0
stop_frequency = 20000.0
sweep = Utilities.generate_frequency_sweep(duration, start_frequency, stop_frequency, speaker_sample_rate, speaker_num_channels)

# Generate pure tone
duration = 5.0
sample_rate = 48000
frequency = 440.0
tone = Utilities.generate_pure_tone(duration, frequency, speaker_sample_rate, speaker_num_channels)

# Initialize speaker
speaker = Speaker.Speaker(output_device, speaker_num_channels, 'int32', speaker_sample_rate, speaker_buffer_size)
speaker.start()

# Initialize microphone
microphone = Microphone.Microphone(input_device, microphone_num_channels, 'int32', microphone_sample_rate, microphone_buffer_size, microphone_max_samples)
microphone.start()

# ------------------------------
# Noise
# ------------------------------

# Wait for noise recording to finish
time.sleep(10.00)

# Store noise recording
noise_recording = np.copy(microphone.latest(microphone.valid_samples))
microphone.save_wav(f"{output_folder}/noise_recording.wav", microphone.valid_samples)

# ------------------------------
# Sweep
# ------------------------------

# Reset recording
microphone.reset()

# Send sound speaker
speaker.write(sweep)

# Wait for output to finish
while speaker.is_playing():
    time.sleep(0.01)

# Store sweep recording
sweep_recording = np.copy(microphone.latest(microphone.valid_samples))
microphone.save_wav(f"{output_folder}/sweep_recording.wav", microphone.valid_samples)

# ------------------------------
# Sweep
# ------------------------------

# Reset recording
microphone.reset()

# Send sound speaker
speaker.write(tone)

# Wait for output to finish
while speaker.is_playing():
    time.sleep(0.01)

# Store tone recording
tone_recording = np.copy(microphone.latest(microphone.valid_samples))
microphone.save_wav(f"{output_folder}/tone_recording.wav", microphone.valid_samples)

# Shutdown
speaker.stop()
microphone.stop()

# ------------------------------
# Normalize and Truncate
# ------------------------------
max_noise = np.max(np.abs(noise_recording))
max_sweep = np.max(np.abs(sweep_recording))
max_tone = np.max(np.abs(tone_recording))
norm_factor = max(max_noise, max_sweep, max_tone)
len_noise  = noise_recording.shape[0]
len_sweep  = sweep_recording.shape[0]
len_tone  = tone_recording.shape[0]
len_min = min(len_noise, len_sweep, len_tone)

# ------------------------------
# Process Noise
# ------------------------------
normalized = (noise_recording / norm_factor)[:len_min,0]
frequencies, times, noise_Sxx = signal.spectrogram(normalized, fs=speaker_sample_rate, nperseg=4096, nfft=8192, noverlap=2048)

# ------------------------------
# Process Sweep
# ------------------------------
normalized = (sweep_recording / norm_factor)[:len_min,0]
frequencies, times, Sxx = signal.spectrogram(normalized, fs=speaker_sample_rate, nperseg=4096, nfft=8192, noverlap=2048)
ideal_frequencies, ideal_times, ideal_Sxx = signal.spectrogram(sweep[:len_min,0], fs=speaker_sample_rate, nperseg=4096, nfft=8192, noverlap=2048)

# Convert power spectrogram to dB scale
Sxx_dB = 10 * np.log10(Sxx + 1e-10)  # Small epsilon to avoid log(0)
ideal_Sxx_dB = 10 * np.log10(ideal_Sxx + 1e-10)  # Small epsilon to avoid log(0)

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

# Remove background noise
spectrogram = Sxx - noise_Sxx

# Average power spectral density across time
average_power = np.mean(spectrogram, axis=1)  # Collapse time axis to get average at each frequency

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

# ------------------------------
# Process Tone
# ------------------------------
normalized = (tone_recording / norm_factor)[:len_min,0]
frequencies, times, Sxx = signal.spectrogram(normalized, fs=speaker_sample_rate, nperseg=4096, nfft=8192, noverlap=2048)

# Convert power spectrogram to dB scale
Sxx_dB = 10 * np.log10(Sxx + 1e-10)  # Small epsilon to avoid log(0)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies, Sxx_dB, shading='gouraud')
plt.colorbar(label="Power (dB)")
plt.title("Spectrogram of Pure Tone")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.ylim(0, speaker_sample_rate / 10)
plt.tight_layout()
plt.savefig(f"{output_folder}/tone_spectrogram.png", dpi=300, bbox_inches="tight")

# Remove background noise
spectrogram = Sxx - noise_Sxx

# Average power spectral density across time
average_power = np.mean(Sxx, axis=1)  # Collapse time axis to get average at each frequency

# Compute THD
fundamental_freq = 440.0
num_harmonics = 5       # Number of harmonics to include in THD calculation
freq_tolerance = 5      # Tolerance in Hz to account for slight shifts

# Find indices for the fundamental and harmonics
fundamental_idx = np.argmin(np.abs(frequencies - fundamental_freq))
harmonic_indices = [
    np.argmin(np.abs(frequencies - (n * fundamental_freq)))
    for n in range(2, num_harmonics + 1)
]

# Compute the power of the fundamental and harmonics
fundamental_power = np.sum(spectrogram[fundamental_idx, :]**2)  # Squared magnitude for power
harmonic_powers = [
    np.sum(Sxx[idx, :]**2) for idx in harmonic_indices
]

# Total Harmonic Distortion
thd = np.sqrt(np.sum(harmonic_powers)) / np.sqrt(fundamental_power)
thd_percentage = thd * 100
print(f"THD: {thd:.4f} ({thd_percentage:.2f}%)")

# Convert to dB scale
average_power_dB = 10 * np.log10(average_power + 1e-10)  # Avoid log(0)

# Plot frequency response curve
plt.figure(figsize=(10, 6))
plt.semilogx(frequencies, average_power_dB, label="Frequency Response Curve", color="b", linewidth=2)
plt.title(f"Frequency Response Curve (Log Scale): THD = {thd_percentage:.4f}%")
plt.xlabel("Frequency [Hz] (Log Scale)")
plt.ylabel("Power (dB)")
plt.grid(which="both", linestyle="--", linewidth=0.5)  # Grid for both major/minor ticks
plt.xlim(20, speaker_sample_rate / 2)  # Limit x-axis to relevant range
plt.tight_layout()
plt.legend()
plt.savefig(f"{output_folder}/tone_response.png", dpi=300, bbox_inches="tight")

# FIN
