# Measure the timing of volume peaks detected on the left vs. right channel
# - This timing difference can contain information about where the sound is coming from
import os
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import NB3.Sound.microphone as Microphone
import NB3.Sound.utilities as Utilities

# Specify paths
username = os.getlogin()
repo_path = f"/home/{username}/NoBlackBoxes/LastBlackBox"
project_path = f"{repo_path}/boxes/audio/signal-processing/python"

# Specify microphone params
input_device = 1
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
for i in range(50):                             # Process 50 buffers (10 per second)
    latest = microphone.latest(buffer_size)     # Get the latest audio buffer
    left_volume = np.mean(np.abs(latest[:,0]))  # Extract left channel volume (abs value of audio signal)
    right_volume = np.mean(np.abs(latest[:,1])) # Extract right channel volume (abs value of audio signal)
    print("{0:.2f} {1:.2f}".format(left_volume, right_volume)) # Print volume level to terminal screen
    time.sleep(0.1) # Wait a bit

# Store full sound recording
recording = np.copy(microphone.sound)

# Shutdown microphone
microphone.stop()

# Compute volume
left_volume = np.abs(recording[:,0])
right_volume = np.abs(recording[:,1])

# Smooth the volume signal
window_len = 100
window = np.ones(window_len) / window_len
left_smooth = np.convolve(left_volume, window, mode='same')
right_smooth = np.convolve(right_volume, window, mode='same')
 
# Find volume peaks
left_peak = np.argmax(left_smooth)      # Get sample with largest value (left channel - smoothed)
right_peak = np.argmax(right_smooth)    # Get sample with largest value (right channel - smoothed)
print(left_peak, right_peak)

# Extract window around peak
padding = 2400
left_window = recording[(left_peak-padding):(left_peak+padding), 0]
left_window_smooth = left_smooth[(left_peak-padding):(left_peak+padding)]
right_window = recording[(right_peak-padding):(right_peak+padding), 1]
right_window_smooth = right_smooth[(right_peak-padding):(right_peak+padding)]

# Now find the time the sound started by searching for first threshold crossing
start_threshold = 0.1
cc = sp.signal.correlate(left_window, right_window, mode='full', method='fft')
lags = sp.signal.correlation_lags(len(left_window), len(right_window), mode='full')
delay = lags[np.argmax(cc)]
print(delay)

# Plot volume peaki(
y_min = np.min([left_window, right_window]) * 1.1
y_max = np.max([left_window, right_window]) * 1.1

plt.figure()
plt.tight_layout()

plt.subplot(3,1,1)
plt.plot(left_window, 'b')
plt.plot(left_window_smooth, 'r')
plt.ylabel("Peak (Left)")
plt.ylim(y_min, y_max)
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(right_window, 'b')
plt.plot(right_window_smooth, 'r')
plt.xlabel("Sample Number")
plt.ylabel("Peak (Right)")
plt.ylim(y_min, y_max)
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(cc)
plt.grid(True)

# Save volume peak timing plot
save_path = f"{project_path}/my_timing_measurement.png"
plt.savefig(f"{save_path}")

#FIN