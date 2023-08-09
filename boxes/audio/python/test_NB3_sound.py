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
microphone = sound.microphone(1, 2, 48000, pyaudio.paInt16, 4800, 480000)
microphone.start()

# Wait to start talking
input("Press Enter to start recording...")

# Start recording
microphone.reset()

# Wait to stop talking
input("Press Enter to stop recording.")

# Read sound recorded
recording = microphone.read()

# Report
print(len(recording))

# Shutdown microphone
microphone.stop()

# Initialize speaker thread
speaker = sound.speaker(1, 2, 48000, pyaudio.paInt16, 4800)
speaker.start()

# Output
speaker.write(recording)

# Wait to stop talking
while speaker.playing():
    time.sleep(0.001)

# Shutdown microphone
speaker.stop()

# FIN