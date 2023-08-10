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
wav_path = box_path + '/_tmp/test.wav'

# List available sound devices
sound.list_devices()

# Initialize microphone thread
microphone = sound.microphone(1, 2, 44100, 4410, 441000, False)
microphone.start()

# Wait to start recording
input("Press Enter to start recording...")

# Start recording
microphone.reset()
microphone.start_recording_wav(wav_path, 441000)

# Wait to stop recording
input("Press Enter to stop recording.")

# Stop recording
microphone.stop_recording_wav()

# Shutdown microphone
microphone.stop()

# FIN