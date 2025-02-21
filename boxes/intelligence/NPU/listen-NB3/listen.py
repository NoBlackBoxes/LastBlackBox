# Import libraries
import numpy as np
import tflite_runtime.interpreter as tflite

# Import modules
import NB3.Sound.microphone as Microphone
import NB3.Sound.utilities as Utilities

# Specify model
model_file = "/home/student/NoBlackBoxes/LastBlackBox/boxes/intelligence/NPU/listen-NB3/_tmp/voice_commands_v0.8_edgetpu.tflite"

# Load delegate (EdgeTPU)
delegate = tflite.load_delegate('libedgetpu.so.1')

# Create interpreter
interpreter = tflite.Interpreter(model_path=model_file, experimental_delegates=[delegate])
interpreter.allocate_tensors()

# Initialize microphone
input_device = 1
num_channels = 2
sample_rate = 48000
buffer_size = int(sample_rate / 10)
max_samples = int(sample_rate * 10)
microphone = Microphone.Microphone(input_device, num_channels, 'int32', sample_rate, buffer_size, max_samples)
microphone.gain = 10.0
microphone.start()

# Compute mel spectrums

# Send to NPU

# Parse results




# Shutdown microphone
microphone.stop()
