import os
import time
import numpy as np

# Locals libs
import NBB_sound.speaker as Speaker
import NBB_sound.utilities as Utilities

# Reimport
import importlib
importlib.reload(Speaker)
importlib.reload(Utilities)

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
box_path = repo_path + '/boxes/audio'
wav_path = box_path + '/_data/sounds/Bach_prelude_C_major.wav'

# Specify params
output_device = 1
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
speaker.play_wav(wav_path)

# Wait for finish
while speaker.is_playing():
    time.sleep(0.1)

# Shutdown speaker
speaker.stop()

# Report
print("Profiling:\n- Avg (Max) Callback Duration (us): {0:.2f} ({1:.2f})".format(speaker.callback_accum/speaker.callback_count*1000000.0, speaker.callback_max*1000000.0))

#FIN