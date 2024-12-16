# Measure Speaker Performance
# - Frequency Response
# - Distortion

# Import libraries
import time
import numpy as np
import matplotlib.pyplot as plt

# Import modules
import NB3.Sound.speaker as Speaker
import NB3.Sound.utilities as Utilities

# Specify speaker params
output_device = 3
num_channels = 2
sample_rate = 48000
buffer_size = int(sample_rate / 10)
max_samples = int(sample_rate * 10)

# List available sound devices
Utilities.list_devices()

## Generate sine wave
#duration = 1.0
#frequency = 1000
#left_data = np.sin(2.0 * np.pi * np.arange(sample_rate*duration) * (frequency / sample_rate))
#right_data = np.sin(2.0 * np.pi * np.arange(sample_rate*duration) * (frequency / sample_rate))
#sound = np.vstack((left_data, right_data)).T

# Generate sine wave sweep (20 Hz to 20,000 Hz)
duration = 5.0
sample_rate = 48000
num_samples = int(sample_rate * duration)
start_frequency = 20.0
stop_frequency = 20000.0
data = np.zeros(num_samples)
phase = 0
frequency = start_frequency
frequency_step = (stop_frequency - start_frequency) / num_samples
phase_step = 2.0 * np.pi * (start_frequency / sample_rate)
for i in range(num_samples):
    data[i] = np.sin(phase)
    phase += phase_step
    frequency += frequency_step
    phase_step = 2.0 * np.pi * (frequency / sample_rate)
sound = np.vstack((data, data)).T

# Initialize speaker
speaker = Speaker.Speaker(output_device, num_channels, 'int32', sample_rate, buffer_size)
speaker.start()

# Send sound speaker
speaker.write(sound)

# Wait for finish
while speaker.is_playing():
    time.sleep(0.1)

# Close
speaker.stop()
print("Done")

# FIN
