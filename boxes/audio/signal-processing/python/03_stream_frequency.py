# Stream the frequency spectrum measured for a short sound buffer using a network socket
import os, pathlib, time
import socket, netifaces
import numpy as np
import NB3.Sound.microphone as Microphone
import NB3.Sound.utilities as Utilities

# Specify paths
repo_path = f"{pathlib.Path.home()}/NoBlackBoxes/LastBlackBox"
project_path = f"{repo_path}/boxes/audio/signal-processing/python"

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

# Specify IP address and port to use for the socket
ip_address = netifaces.ifaddresses("wlan0")[netifaces.AF_INET][0]['addr'] # Get IP address for WiFi interface
port = 1234

# Create a Socket that listens for connections
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # AF_INET means use IPv4 address and SOCK_STREAM means use TCP (rather than UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Tell the socket to reuse this address (e.g. if it crashes)
sock.bind((ip_address, port)) # Bind the new socket to the specified IP address and port
sock.listen() # Start listening for connections
print(f"Socket Server listening on {ip_address}:{port}")

# Determine frequency output range of FFT
freqs = np.fft.fftfreq(buffer_size, d=1/sample_rate) # Compute the frequency bins based on sample rate (48 kHz) and length of buffer (100 ms)
max_freq = sample_rate / 2                           # Max frequency is 24 kHz (Nyquist Criteria)
frequency_range = (freqs >= 50) * (freqs <= 5000)    # Only consider frequencies between 50 Hz and 5 kHz
freqs = freqs[frequency_range]                       # Keep only that frequency range
print(f"Samples in each spectrum: {len(freqs)}")     # Report number of samples per spectrum

# The Socket Server Loop(s)
samples_per_buffer = len(freqs)
try:
    while True:                             # This loop will keep checking for a connection
        conn, addr = sock.accept()          # Accept a connection request (waits until one is received)
        print(f"Connected to by {addr}")

        try:
            while True:
                latest = microphone.latest(buffer_size)     # Get latest sound input
                channel = latest[:,0]                       # Only keep the LEFT channel
                fft = np.fft.fft(channel)                   # FFT
                fft = np.abs(fft[frequency_range])          # Keep frequencies from 50 HZ to 5 kHz
                fft = fft / np.max(fft)                     # Normalize
                fft = fft.astype(np.float32)                # Convert to Float32
                conn.sendall(fft)                           # Send data to socket
                time.sleep(0.05)

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            print("Client disconnected.")
        finally:
            conn.close()
            print("Connection closed; returning to accept new clients.")

except KeyboardInterrupt:
    print("\nShutting down...")

finally:
        sock.close()
        microphone.stop()

#FIN