import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Locals libs
import NB3.Sound.microphone as Microphone
import NB3.Sound.utilities as Utilities

# Reimport
import importlib
importlib.reload(Microphone)
importlib.reload(Utilities)

# Get user name
username = os.getlogin()

# Specify paths
repo_path = f"/home/{username}/NoBlackBoxes/LastBlackBox"
project_path = f"{repo_path}/boxes/audio/signal-processing/python"

# Specify params
input_device = 3
num_channels = 2
sample_rate = 48000
buffer_size = int(sample_rate / 10)
max_samples = int(sample_rate * 5)

# List available sound devices
Utilities.list_devices()

# Initialize microphone
microphone = Microphone.Microphone(input_device, num_channels, 'int32', sample_rate, buffer_size, max_samples)
microphone.gain = 10.0
microphone.start()

# Clear error ALSA/JACK messages from terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Wait to save recording
input("Press <Enter> to start 5 second recording...")

# Live volume processing
for i in range(50): # 50 buffers (10 per second)
    latest = microphone.latest(buffer_size)
    if num_channels == 2:
        left_volume = np.mean(np.abs(latest[:,0]))
        right_volume = np.mean(np.abs(latest[:,1]))
        print("{0:.2f} {1:.2f}".format(left_volume, right_volume))
    else:
        volume = np.mean(np.abs(latest[:]))
        print("{0:.2f}".format(volume))
    time.sleep(0.1)

# Store recording
recording = np.copy(microphone.sound)

# Compute volume
volume = np.abs(recording)

# Shutdown microphone
microphone.stop()

# Plot volume recording
plt.figure()
plt.subplot(2,1,1)
plt.plot(volume[:,0])
plt.subplot(2,1,2)
plt.plot(volume[:,1])

# Save volume recording
save_path = f"{project_path}/my_volume_measurement.png"
plt.savefig(f"{save_path}")

#FIN