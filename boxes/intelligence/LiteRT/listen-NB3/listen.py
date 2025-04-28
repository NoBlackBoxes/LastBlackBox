# Imports
import os
import time
import curses
import serial
import numpy as np
import tflite_runtime.interpreter as tflite
import NB3.Sound.microphone as Microphone
import NB3.Sound.utilities as Utilities

# Get user name
username = os.getlogin()

# Set base path
base_path = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/intelligence/LiteRT"

# Configure serial port
ser = serial.Serial()
ser.baudrate = 115200
ser.port = '/dev/ttyUSB0'
ser.open()

# Specify model and labels
model_path = f"{base_path}/listen-NB3/model/voice_commands_v0.8_edgetpu.tflite"
labels_path = f"{base_path}/listen-NB3/model/labels.txt"

# Create interpreter
interpreter = tflite.Interpreter(model_path=model_path)
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
        best_voice_command = labels[top_3_indices[0]]

        # Build a readable string for top 3 predictions
        for i, index in enumerate(top_3_indices):
            score = output_data[index]
            results = f"[{labels[index]}: {score:.3f}]"
            screen.addstr(i+1, 0, f"  {i}: {results}")
        
        # Respond to commands
        # ADD YOUR COMMAND RESPONSES AFTER HERE ------->
        if best_voice_command == "turn_left":  # If the "best" voice command detected is "turn_left"
            ser.write(b'l')                    # Send the Arduino 'l' (the command to start turing left)  
            time.sleep(1.0)                    # Wait (while moving) for 1 second
            ser.write(b'x')                    # Send the Arduino 'x' (the command to stop)
        # <------- ADD YOUR COMMAND BEFORE RESPONSES HERE
        
        # Add quit instructions
        screen.addstr(5, 0, f"    - Press 'q' to Quit")
        screen.addstr(6, 0, f"")

        # Wait a bit
        time.sleep(0.05)

finally:
    # Cleanup terminal
    curses.nocbreak()
    screen.keypad(0)
    curses.echo()
    curses.endwin()

    # Shutdown microphone
    microphone.stop()
    
    # Close serial port
    ser.close()

#FIN