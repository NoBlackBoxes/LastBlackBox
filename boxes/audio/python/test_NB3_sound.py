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

# Initiliaze microphone thread
sound.list_devices()

# Initiliaze microphone thread
microphone = sound.microphone(1, 4800, pyaudio.paInt16, 1, 48000, 10)
microphone.start()

# Initiliaze speaker thread
speaker = sound.speaker(1, 4800, pyaudio.paInt16, 1, 48000)
speaker.start()

# Wait to start talking
input("Press Enter to start recording...")

# Start recording
microphone.reset()

# Wait to stop talking
input("Press Enter to stop recording and start playback.")

# Read sound recorded
recording = microphone.read()

# Report
print(len(recording))

# Output
speaker.write(recording)

# Wait to stop talking
input("Press Enter to stop playback.")

# Shutdown
microphone.stop()
speaker.stop()

# FIN