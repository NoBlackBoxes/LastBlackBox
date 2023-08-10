import os
import time
import pyaudio
import numpy as np

# Locals libs
import libs.NB3_sound as sound

# Reimport
import importlib
importlib.reload(sound)

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/LastBlackBox'
box_path = repo_path + '/boxes/audio'
wav_path = box_path + '/_data/sounds/Bach_prelude_C_major.wav'

# List available sound devices
sound.list_devices()

# Initialize speaker thread
speaker = sound.speaker(1, 2, 44100, 4410)
speaker.start()

# Output
speaker.play_wav(wav_path)

# Wait for playback to finish
while speaker.is_playing():
    time.sleep(0.01)

# Shutdown speaker
speaker.stop()

# FIN