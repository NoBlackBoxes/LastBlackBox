import os, pathlib, time
import numpy as np
import NB3.Sound.speaker as Speaker
import NB3.Sound.utilities as Utilities

# Specify paths
repo_path = f"{pathlib.Path.home()}/NoBlackBoxes/LastBlackBox"
box_path = repo_path + '/boxes/audio'
wav_path = box_path + '/_resources/sounds/Bach_prelude_C_major.wav'

# Get speaker device by name (NB3: "MAX", PC: select based on listed output devices)
output_device = Utilities.get_output_device_by_name("HD-Audio")
if output_device == -1:
    exit("Output device not found")

# Specify params
num_channels = 2
sample_rate = 48000
buffer_size = int(sample_rate / 10)
max_samples = int(sample_rate * 10)

# List available sound devices
Utilities.list_devices()

# Initialize speaker
speaker = Speaker.Speaker(output_device, num_channels, 'int32', sample_rate, buffer_size)
speaker.start()

# Clear error ALSA/JACK messages from terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Play WAV file
sound = speaker.play_wav(wav_path)

# Wait for finish
while speaker.is_playing():
    time.sleep(0.1)

# Shutdown speaker
speaker.stop()

# Report
print("Profiling:\n- Avg (Max) Callback Duration (us): {0:.2f} ({1:.2f})".format(speaker.callback_accum/speaker.callback_count*1000000.0, speaker.callback_max*1000000.0))

#FIN