import time
import numpy as np
import NB3.Sound.microphone as Microphone
import NB3.Sound.utilities as Utilities
import NB3.Plot.line as Line

# Specify params
input_device = 0
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

# Open line plot
line = Line.Line(512, 256, -1.0, 1.0, num_samples=buffer_size)
line.open()

# your blocking loop (e.g., socket recv in same thread)
try:
    start_time = time.time()
    prev = np.zeros(buffer_size, np.float32)
    while True:
        latest = microphone.latest(buffer_size)
        if len(latest) < buffer_size:
            print(len(latest))
            continue
        data = latest[:,0]
        #print(prev-data)
        line.draw_data(data)      # push data
        line.process_events()     # handle window events
        line.render()             # draw immediately
        time.sleep(0.02)
        end_time = time.time()
        print(end_time - start_time)
        volume = np.mean(np.max(data))
        print("{0:.2f}".format(volume))
        start_time = end_time
        prev = np.copy(data)
        print("Profiling:\n- Avg (Max) Callback Duration (us): {0:.2f} ({1:.2f})".format(microphone.callback_accum/microphone.callback_count*1000000.0, microphone.callback_max*1000000.0))

finally:
    microphone.stop()
    line.close()

# FIN