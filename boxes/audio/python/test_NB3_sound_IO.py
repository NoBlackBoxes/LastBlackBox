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
box_path = repo_path + '/boxes/audio/python'

# List available sound devices
sound.list_devices()

# Initialize microphone thread
microphone = sound.microphone(4, 2, 44100, 4410, 441000, False)
microphone.start()

# Initialize speaker thread
speaker = sound.speaker(4, 2, 44100, 441)
speaker.start()

# Wait to start recording
input("Press Enter to start recording...")

# Start recording
microphone.reset()

# Wait to stop recording
input("Press Enter to stop recording.")

# Read sound recorded
recording = microphone.read()

# Report
print(len(recording))

# Output
speaker.write(recording)

# Wait for playback to finish
while speaker.is_playing():
    time.sleep(0.01)

# Shutdown speaker
speaker.stop()
microphone.stop()

# FIN