import os
import time
import socket
import netifaces
import numpy as np

# Locals libs
import NB3.Sound.microphone as Microphone
import NB3.Sound.utilities as Utilities

# Reimport
import importlib
importlib.reload(Microphone)
importlib.reload(Utilities)

# Get user name
username = os.getlogin()

# Specify paths
repo_path = f"/home/{username}/NoBlackBoxes/LastBlackBox"
project_path = f"{repo_path}/boxes/audio/signal-processing/python"

# Specify params
input_device = 1
num_channels = 2
sample_rate = 48000
buffer_size = int(sample_rate / 10)
max_samples = int(sample_rate * 5)

# List available sound devices
Utilities.list_devices()

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

# The Socket Server Loop(s)
freqs = np.fft.fftfreq(buffer_size, d=1/sample_rate)[:buffer_size // 2] # Get a list of frequencies that will result from FFT (in steps of 10 Hz)
freqs = freqs[5:500] # Only look at frequencies between 50 Hz and 5 kHz
samples_per_buffer = len(freqs)
print(f"Samples in each streamed buffer: {samples_per_buffer}")
try:
    while True:                             # This loop will keep checking for a connection
        conn, addr = sock.accept()          # Accept a connection request (waits until one is received)
        print(f"Connected to by {addr}")

        try:
            while True:
                latest = microphone.latest(buffer_size)     # Get latest sound input
                channel = latest[:,0]                       # Only use LEFT channel
                fft = np.fft.fft(channel)                   # FFT
                fft = np.abs(fft[5:500])                    # Positive frequencies from 50 HZ to 5 kHz
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