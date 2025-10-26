import time
import numpy as np
import NB3.Sound.microphone as Microphone
import NB3.Sound.utilities as Utilities
import NB3.Plot.line as Line

# Specify params
input_device = 3
num_channels = 1
sample_rate = 48000
buffer_size = int(sample_rate / 100)
max_samples = int(sample_rate * 10)

# List available sound devices
Utilities.list_devices()

# Initialize microphone
microphone = Microphone.Microphone(input_device, num_channels, 'int32', sample_rate, buffer_size, max_samples)
microphone.gain = 1.0
microphone.start()

# Open line plotter
line = Line.Line(-1.1, 1.1, num_samples=buffer_size*100)
line.open()

# Acquisition loop
while True:
    try:
        start_time = time.time()
        latest = microphone.latest(-1) # Grab latest "new" samples
        data = latest[:,0]             # Only take Channel 0
        line.plot(data)                # Plot data

        # Profiling
        end_time = time.time()
        volume = np.mean(np.max(data))
        start_time = end_time
        #print("Profiling:\n- Avg (Max) Callback Duration (us): {0:.2f} ({1:.2f})".format(microphone.callback_accum/microphone.callback_count*1000000.0, microphone.callback_max*1000000.0))
        time.sleep(0.05)

    except KeyboardInterrupt:
        break

# Cleanup
microphone.stop()
line.close()

#FIN