import os, pathlib, time
import numpy as np
import NB3.Sound.microphone as Microphone
import NB3.Sound.speaker as Speaker
import NB3.Sound.utilities as Utilities

# Specify paths
repo_path = f"{pathlib.Path.home()}/NoBlackBoxes/LastBlackBox"
box_path = repo_path + '/boxes/audio'
wav_path = repo_path + '/_tmp/test.wav'

# Specify params
input_device = 1
output_device = 1
num_input_channels = 2
num_output_channels = 2
sample_rate = 48000
buffer_size = int(sample_rate / 100)
max_samples = int(sample_rate * 10)

# List available sound devices
Utilities.list_devices()

# Initialize microphone
microphone = Microphone.Microphone(input_device, num_input_channels, 'int16', sample_rate, buffer_size, max_samples)
microphone.start()

# Initialize speaker
speaker = Speaker.Speaker(output_device, num_output_channels,  'int16', sample_rate, buffer_size)
speaker.start()

# Clear error ALSA/JACK messages from terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Live input/output (FEEDBACK!)
for i in range(1000):
    latest = microphone.latest(buffer_size)
    speaker.write(latest)
    time.sleep(0.01)

# Shutdown microphone
microphone.stop()

# Shutdown speaker
speaker.stop()


#FIN