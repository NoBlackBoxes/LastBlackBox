# Imports
import os
import time
import curses
import serial
import torch
import numpy as np
import NB3.Sound.microphone as Microphone
import NB3.Sound.utilities as Utilities

# Set base path
username = os.getlogin()
base_path = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/intelligence/pytorch/keyword_spotter"

# Specify model and labels
model_path = f"{base_path}/_tmp/quantized/quantized.pt"
labels_path = f"{base_path}/labels.txt"

# Configure serial port
ser = serial.Serial()
ser.baudrate = 115200
ser.port = '/dev/ttyUSB0'
ser.open()

# Reload saved quantized model
torch.backends.quantized.engine = 'qnnpack'
quantized_model = torch.jit.load(model_path, map_location=torch.device('cpu'))
quantized_model.eval()

# Limit CPU resources
torch.set_num_threads(2)

# Load labels
labels = np.genfromtxt(labels_path, dtype=str)

# Initialize microphone
input_device = 1
num_channels = 1
sample_rate = 16000
buffer_size = int(sample_rate / 10)
max_samples = int(sample_rate * 10)
microphone = Microphone.Microphone(input_device, num_channels, 'int32', sample_rate, buffer_size, max_samples)
microphone.gain = 100.0
microphone.start()

# Generate Mel Matrix (for audio processing)
mel_matrix = Utilities.generate_mel_matrix(16000, 40) # 40 Mel Coeffs

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
        latest = microphone.latest(sample_rate)[:,0]
        if len(latest) < sample_rate:
            continue
        mel_spectrogram = Utilities.compute_mel_spectrogram(latest, 640, 320, mel_matrix)

        # Clear screen
        screen.erase()

        # Are we waiting for sufficient audio in the buffer?
        if(mel_spectrogram is None):
            screen.addstr(0, 0, 'Status: ...filling buffer...')       
            time.sleep(0.1)
            continue
        screen.addstr(0, 0, 'Status: ...Listening...')       

        # Convert ndarray to Tensor
        features = np.expand_dims(mel_spectrogram, 0)
        features_tensor = torch.from_numpy(features).to(dtype=torch.float32).unsqueeze(0)
        features_tensor = features_tensor.to('cpu')

        # Run model
        output = quantized_model(features_tensor).detach().cpu()[0]
        probabilities = torch.nn.functional.softmax(output, dim=0).numpy()

        # Get indices of top 3 predictions
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        best_voice_command = labels[top_3_indices[0]]
        best_voice_score = probabilities[top_3_indices[0]]

        # Build a readable string for top 3 predictions
        for i, index in enumerate(top_3_indices):
            score = probabilities[index]
            results = f"[{labels[index]}: {score:.3f}]"
            screen.addstr(i+1, 0, f"  {i}: {results}")
            screen.addstr(i+1, 0, f"  {i}: {results}")

        # Add quit instructions
        screen.addstr(5, 0, f"    - Press 'q' to Quit")
        screen.addstr(6, 0, f"")
        screen.refresh()

        # Respond to commands
        # ADD YOUR COMMAND RESPONSES AFTER HERE ------->
        if best_voice_score > 0.8:
            if best_voice_command == "left":  # If the "best" voice command detected is "left"
                ser.write(b'l')                    # Send the Arduino 'l' (the command to start turning left)  
                time.sleep(1.0)                    # Wait (while moving) for 1 second
                ser.write(b'x')                    # Send the Arduino 'x' (the command to stop)
        # <------- ADD YOUR COMMAND BEFORE RESPONSES HERE
        
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