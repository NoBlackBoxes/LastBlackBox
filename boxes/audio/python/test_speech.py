import os
import time
import numpy as np

# Locals libs
import NBB_sound.microphone as Microphone
import NBB_sound.utilities as Utilities

# Reimport
import importlib
importlib.reload(Microphone)
importlib.reload(Utilities)

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
box_path = repo_path + '/boxes/audio'
wav_path = repo_path + '/_tmp/test.wav'

# Specify params
input_device = 0
num_channels = 1
sample_rate = 16000
buffer_size = int(sample_rate / 10)
max_samples = int(sample_rate * 10)

# List available sound devices
Utilities.list_devices()

# Initialize microphone
microphone = Microphone.Microphone(input_device, num_channels, 'int16', sample_rate, buffer_size, max_samples)
microphone.start()

# Clear error ALSA/JACK messages from terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Live speech detection
for i in range(100):
    if microphone.is_speech():
        print("speech")
    else:
        print("-not-")
    time.sleep(0.1)

# Shutdown microphone
microphone.stop()

# Report
print("Profiling:\n- Avg (Max) Callback Duration (us): {0:.2f} ({1:.2f})".format(microphone.callback_accum/microphone.callback_count*1000000.0, microphone.callback_max*1000000.0))

# FIN