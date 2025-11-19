# Measure the volume of sound recorded by the microphone
import os, pathlib, time
import numpy as np
import matplotlib.pyplot as plt
import NB3.Sound.microphone as Microphone
import NB3.Sound.utilities as Utilities

# Specify paths
repo_path = f"{pathlib.Path.home()}/NoBlackBoxes/LastBlackBox"
project_path = f"{repo_path}/boxes/audio/signal-processing/python"

# List available sound devices
Utilities.list_devices()

# Get microphone device by name (NB3: "MAX", PC: select based on listed input devices)
input_device = Utilities.get_input_device_by_name("HD-Audio")
if input_device == -1:
    exit("Input device not found")

# Specify microphone params
num_channels = 2
sample_rate = 48000
buffer_size = int(sample_rate / 10)
max_samples = int(sample_rate * 5)

# Initialize microphone
microphone = Microphone.Microphone(input_device, num_channels, 'int32', sample_rate, buffer_size, max_samples)
microphone.gain = 10.0
microphone.start()

# Clear error ALSA/JACK messages from terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Wait to save recording
input("Press <Enter> to start 5 second recording...")

# Live volume processing
Utilities.meter_start()
for i in range(50):                                     # Process 50 buffers (10 per second)
    latest = microphone.latest(buffer_size)             # Get the latest audio buffer
    left_volume = np.mean(np.abs(latest[:,0]))          # Extract left channel volume (abs value of audio signal)
    right_volume = np.mean(np.abs(latest[:,1]))         # Extract right channel volume (abs value of audio signal)
    Utilities.meter_update(left_volume, right_volume)   # Update volume meter
    time.sleep(0.1) # Wait a bit
Utilities.meter_stop()

# Store full sound recording
recording = np.copy(microphone.sound)

# Shutdown microphone
microphone.stop()

# Compute volume
volume = np.abs(recording)

# Plot volume recording
plt.figure()
plt.tight_layout()

plt.subplot(2,1,1)
plt.plot(volume[:,0])
plt.ylabel("Volume (Left)")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(volume[:,1])
plt.xlabel("Sample Number")
plt.ylabel("Volume (Right)")
plt.grid(True)

# Save volume plot
save_path = f"{project_path}/my_volume_measurement.png"
plt.savefig(f"{save_path}")

#FIN