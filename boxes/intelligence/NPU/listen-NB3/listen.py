# Import libraries
import os
import time
import numpy as np
import tflite_runtime.interpreter as tflite

# Import modules
import NB3.Sound.microphone as Microphone
import NB3.Sound.utilities as Utilities

# Get user name
username = os.getlogin()

# Specify model
model_file = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/intelligence/NPU/listen-NB3/_tmp/voice_commands_v0.8_edgetpu.tflite"

# Load delegate (EdgeTPU)
delegate = tflite.load_delegate('libedgetpu.so.1')

# Create interpreter
interpreter = tflite.Interpreter(model_path=model_file, experimental_delegates=[delegate])
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize microphone
input_device = 1
num_channels = 1
sample_rate = 16000
buffer_size = int(sample_rate / 10)
max_samples = int(sample_rate * 10)
microphone = Microphone.Microphone(input_device, num_channels, 'int32', sample_rate, buffer_size, max_samples)
microphone.gain = 100.0
microphone.start()

# Live processing
for i in range(50):

    # Compute mel features
    mel_spectrogram = microphone.mel_spectrogram()

    # Are we waiting for sufficient audio in the buffer?
    if(mel_spectrogram is None):
        print("...filling buffer")
        time.sleep(0.1)
        continue
    
    # Normalize spectrogram
    mel_spectrogram -= np.mean(mel_spectrogram, axis=1, keepdims=True)
    mel_spectrogram /= (3 * np.std(mel_spectrogram, axis=1, keepdims=True))
    mel_spectrogram += 1
    mel_spectrogram *= 127.5
    mel_spectrogram = np.clip(mel_spectrogram, 0, 255).astype(np.uint8)

    # Send to NPU
    interpreter.set_tensor(input_details[0]['index'], mel_spectrogram)    

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Get indices of top 3 predictions
    top_3_indices = np.argsort(output_data)[-3:][::-1]

    # Build a readable string for top 3 predictions
    top_3_results = []
    for index in top_3_indices:
        score = output_data[index]
        top_3_results.append(f"[{index}: {score:.3f}]")

    # Print all on one line
    print("Top 3 predictions:", ", ".join(top_3_results))
    
    # Wait a bit
    time.sleep(0.1)

microphone.save_wav("_tmp/here.wav", microphone.valid_samples)

# Shutdown microphone
microphone.stop()
