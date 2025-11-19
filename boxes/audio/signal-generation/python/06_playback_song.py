# Playback a song stored in a WAV file 
import os, pathlib, time
import numpy as np
import matplotlib.pyplot as plt
import NB3.Sound.speaker as Speaker
import NB3.Sound.utilities as Utilities

# Specify paths
repo_path = f"{pathlib.Path.home()}/NoBlackBoxes/LastBlackBox"
project_path = f"{repo_path}/boxes/audio/signal-generation/python"
song_path = f"{repo_path}/boxes/audio/_resources/sounds/Bach_prelude_C_major.wav"
#song_path = f"{repo_path}/boxes/audio/_resources/sounds/Axel_F.wav"

# List available sound devices
Utilities.list_devices()

# Get speaker device by name (NB3: "MAX", PC: select based on listed output devices)
output_device = Utilities.get_output_device_by_name("HD-Audio")
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

# Wait to save recording
input("Press <Enter> to start playback...")

# Play WAV file
song = speaker.play_wav(song_path)

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
save_path = f"{project_path}/my_wav_spectrogram_measurement.png"
plt.savefig(f"{save_path}")

#FIN