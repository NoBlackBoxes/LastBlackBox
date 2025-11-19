import os, pathlib, time
import numpy as np
import matplotlib.pyplot as plt
import NB3.Sound.microphone as Microphone
import NB3.Sound.utilities as Utilities

# Specify paths
repo_path = f"{pathlib.Path.home()}/NoBlackBoxes/LastBlackBox"
tmp_path = repo_path + '/_tmp/sounds'
wav_path = tmp_path + '/test.wav'

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

# Save recording
microphone.save_wav(wav_path, sample_rate*3)

# Store full sound recording
recording = np.copy(microphone.sound)

# Shutdown microphone
microphone.stop()

# Report
print("Profiling:\n- Avg (Max) Callback Duration (us): {0:.2f} ({1:.2f})".format(microphone.callback_accum/microphone.callback_count*1000000.0, microphone.callback_max*1000000.0))

# Save plot of recording
plt.figure()
plt.plot(recording)
save_path = wav_path.replace("wav", "png")
plt.savefig(f"{save_path}")

#FIN