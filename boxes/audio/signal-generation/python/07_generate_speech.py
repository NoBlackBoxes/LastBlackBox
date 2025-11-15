# Generate speech using classic methods for Text-to-Speech
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

# Wait to save recording
input("Press <Enter> to start speech generation...")

# Generate speech and send to speaker
speech = speaker.speak("Go, Stop, Go, Stop!")

# Wait for sound output to finish
try:
    while speaker.is_playing():
        time.sleep(0.01)
finally:
    speaker.stop()

# -----------------------------------
# Plot spectrum (using FFT) of speech
# -----------------------------------

# Compute spectrogram
times, frequencies, magnitudes_db = Utilities.compute_spectrogram(speech, sample_rate)

# Plot
plt.figure(figsize=(8, 4))
plt.pcolormesh(times, frequencies, magnitudes_db, shading='gouraud', cmap='plasma')
plt.ylim(10, 10000) # Limit plot range to 10 Hz → 10 kHz
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Speech Spectrogram (10 Hz to 10 kHz)')
plt.colorbar(label='Magnitude [dB]')
plt.tight_layout()

# Save frequency spectrum plot
save_path = f"{project_path}/my_speech_spectrogram_measurement.png"
plt.savefig(f"{save_path}")

#FIN