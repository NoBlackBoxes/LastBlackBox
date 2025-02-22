# Import libraries
import os
import time
import curses
import serial
import numpy as np
import tflite_runtime.interpreter as tflite

# Import modules
#import NB3.Sound.microphone as Microphone
#import NB3.Sound.utilities as Utilities

# Get user name
username = os.getlogin()

# Configure serial port
ser = serial.Serial()
ser.baudrate = 115200
ser.port = '/dev/ttyUSB0'
ser.open()

# Specify model and labels
#model_path = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/intelligence/NPU/look-NB3/model/Posenet-Mobilenet-Quantized_edgetpu.tflite"
#model_path = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/intelligence/NPU/look-NB3/model/posenet_mobilenet_v1_075_324_324_16_quant_decoder_edgetpu.tflite"
model_path = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/intelligence/NPU/look-NB3/model/movenet_single_pose_thunder_ptq_edgetpu.tflite"
#model_path = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/intelligence/NPU/listen-NB3/model/voice_commands_v0.8_edgetpu.tflite"
#labels_path = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/intelligence/NPU/listen-NB3/model/labels.txt"

# Load delegate (EdgeTPU)
delegate = tflite.load_delegate('libedgetpu.so.1')

# Create interpreter
interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=[delegate])
interpreter.allocate_tensors()

# Load labels
labels = np.genfromtxt(labels_path, dtype=str)
labels = np.insert(labels, 0, "-silence-")

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

# Initialize interactive terminal
screen = curses.initscr()
curses.noecho()
curses.cbreak()
screen.keypad(True)
screen.nodelay(True)

# Processing loop
try:
    while True:
        # Check for quit ('q') key
        char = screen.getch()
        if char == ord('q'):
            break

        # Compute mel spectrogram
        mel_spectrogram = microphone.mel_spectrogram()

        # Clear screen
        screen.erase()

        # Are we waiting for sufficient audio in the buffer?
        if(mel_spectrogram is None):
            screen.addstr(0, 0, 'Status: ...filling buffer...')       
            time.sleep(0.1)
            continue
        screen.addstr(0, 0, 'Status: ...Listening...')       
                
        # Send to NPU
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(mel_spectrogram, axis=0))

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Get indices of top 3 predictions
        top_3_indices = np.argsort(output_data)[-3:][::-1]

        # Build a readable string for top 3 predictions
        for i, index in enumerate(top_3_indices):
            score = output_data[index]
            results = f"[{labels[index]}: {score:.3f}]"
            screen.addstr(i+1, 0, f"  {i}: {results}")
        
        # Check for commands
        if labels[top_3_indices[0]] == "turn_left":
            ser.write(b'l')
            time.sleep(2.0)
            ser.write(b'x')
        
        # Wait a bit
        time.sleep(0.05)

finally:
    # Shutdown microphone
    microphone.stop()

    # Close serial port
    ser.close()

    # Cleanup terminal
    curses.nocbreak()
    screen.keypad(0)
    curses.echo()
    curses.endwin()

#FIN