import socket
import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from collections import deque


class RollingBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def addElement(self, element):
        self.buffer.append(element)

    def extendBuffer(self, elements):
        self.buffer.extend(elements)

    def getBuffer(self):
        return list(self.buffer)
    
# Main computer IP address and port number
HOST_IP = ''
HOST_PORT = 5005


CHUNK_SIZE = 4096  # Buffer size
RATE = 22050  # Sample rate (Hz)

def receive_audio():
    # Create a socket and bind it to the host IP and port
    host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_socket.bind((HOST_IP, HOST_PORT))
    host_socket.listen(1)

    # Accept the client connection
    client_socket, _ = host_socket.accept()

    # Create a buffer of audio data
    y_values = RollingBuffer(RATE)  # save one second of audio data
    x_values = []

    # Create two subplots for audio and fourier data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    fig.tight_layout(pad=3.0)
    
    # Create two empty plots
    line_audio, = ax1.plot(x_values, y_values.getBuffer())
    line_fft, = ax2.plot([], [])

    # Set up the plot for audio data
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Audio Signal')
    ax1.set_title('Real-Time Audio Data')
    ax1.set_xlim(0, RATE)
    ax1.set_ylim(-15_000, +15_000)

    # Set up the plot for fourier data
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Real-Time Fourier Transform')
    ax2.set_xlim(0, 10000)
    ax2.set_ylim(0, 5_000_000)
    
    try:
        while True:
            # Receive audio data from the client
            raw_data = client_socket.recv(CHUNK_SIZE)

            interleaved_data = np.frombuffer(raw_data, dtype=np.int16)

            # Extract left and right values
            left = interleaved_data[::2] 
            #right = interleaved_data[1::2]  
            
            # We continue only with the values from one ear from here on
            left = left.flatten()
            
            # Extend the current audio values to the buffer
            y_values.extendBuffer(left)

            # Update audio data plot
            current_y_values = y_values.getBuffer()
            x_values = np.arange(0, len(current_y_values), 1)
            line_audio.set_data(x_values, current_y_values)

            # Update fourier data plot
            fourier = np.abs(rfft(current_y_values))
            freqs = rfftfreq(len(current_y_values), 1 / RATE)
            line_fft.set_data(freqs, fourier)

            # Redraw the plot
            fig.canvas.draw()

            # Pause for a short interval to control the refresh rate
            plt.pause(0.001)

    except KeyboardInterrupt:
        # Close the socket when interrupted
        host_socket.close()

if __name__ == '__main__':
    receive_audio()