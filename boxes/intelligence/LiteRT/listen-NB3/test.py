# Imports
import os
import time
import numpy as np
import scipy as sp 
import curses
import tflite_runtime.interpreter as tflite
import NB3.Sound.microphone as Microphone
import NB3.Sound.utilities as Utilities

# Generate mel matrix
def generate_mel_matrix():
    sample_rate = 16000
    mel_fft_length = 512
    mel_num_bins = 40

    mel_matrix = np.zeros((mel_fft_length // 2 + 1, mel_num_bins))
    freq_bins = np.linspace(0, sample_rate / 2, mel_fft_length // 2 + 1)
    freq_bins_mel = 1127.0 * np.log(1.0 + freq_bins / 700.0)
    mel_bins = np.linspace(1127.0 * np.log(1.0 + 60 / 700.0), 1127.0 * np.log(1.0 + 3800 / 700.0), mel_num_bins + 2)

    for i in range(mel_num_bins):
        lower = mel_bins[i]
        center = mel_bins[i + 1]
        upper = mel_bins[i + 2]
        mel_matrix[:, i] = np.maximum(0, np.minimum((freq_bins_mel - lower) / (center - lower), (upper - freq_bins_mel) / (upper - center)))

    return mel_matrix

# Process sound
def process_sound(sound, mel_matrix=None):
    # Assumes 16000 samples at 16 kHz (1 second) of audio (1 channel)
    # Float32, -1.0 to 1.0
    num_samples = 16000
    sample_rate = 16000

    # Parameters
    mel_window_length_samples = 640     # 40 ms
    mel_hop_length_samples = 320        # 20 ms
    mel_fft_length = 512
    if mel_matrix is None:
        mel_matrix = generate_mel_matrix()

    # Compute spectrogram
    frames = []
    for i in range(0, 16000 - mel_window_length_samples + 1, mel_hop_length_samples):
        frame = sound[i:i+mel_window_length_samples]
        windowed = frame * np.hanning(mel_window_length_samples)
        frames.append(np.abs(np.fft.rfft(windowed, mel_fft_length)))
    spectrogram = np.stack(frames)

    # Apply mel filters and take log
    mel_spectrogram = np.dot(spectrogram, mel_matrix)
    #log_mel_spectrogram = np.log(mel_spectrogram + 0.001)

    # Normalise
    mel_spectrogram -= np.mean(mel_spectrogram, axis=0, keepdims=True)
    mel_spectrogram /= (3 * np.std(mel_spectrogram, axis=0, keepdims=True))

    return np.float32(mel_spectrogram.T)

# Get user name
username = os.getlogin()

# Set base path
base_path = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/intelligence/LiteRT/listen-NB3"

# Specify model and labels
model_path = f"{base_path}/_tmp/custom.tflite"
labels_path = f"{base_path}/_tmp/labels.txt"

# Create interpreter
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Load labels
labels = np.genfromtxt(labels_path, dtype=str)

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)

# Initialize microphone
input_device = 3
num_channels = 1
sample_rate = 48000
buffer_size = int(sample_rate / 10)
max_samples = int(sample_rate * 10)
Utilities.list_devices()
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
        latest = microphone.latest(sample_rate)
        if len(latest) < sample_rate:
            continue
        resampled = sp.signal.resample_poly(latest, up=1, down=3)[:,0] # Resample to 16 kHz
        mel_spectrogram = process_sound(resampled)
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)

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
    
#FIN