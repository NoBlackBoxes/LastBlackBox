import socket
import numpy as np
import struct
import matplotlib.pyplot as plt
from collections import deque


class RollingBuffer:
    """
    A class to implement a rolling buffer of a fixed size.
    """
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def addElement(self, element):
        self.buffer.append(element)

    def getBuffer(self):
        return list(self.buffer)


# Define the host IP address and port number
HOST_IP = ''  # Empty string means to listen on all interfaces
HOST_PORT = 5005


def receive_data():
    # Create a socket and bind it to the host IP and port
    host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_socket.bind((HOST_IP, HOST_PORT))
    host_socket.listen(1)

    # Accept the client connection
    client_socket, _ = host_socket.accept()

    # Create a buffer of audio data
    y_values = RollingBuffer(100)  # save the last 100 values
    x_values = []

    # Create two subplots for audio and fourier data
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.tight_layout(pad=3.0)
    
    # Create two empty plots
    line_audio, = ax.plot(x_values, y_values.getBuffer())

    # Set up the plot for the sensor data
    ax.set_xlabel('Time')
    ax.set_ylabel('Sensor Value')
    ax.set_title('Real-Time Sensor Data')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1500)

    try:
        while True:
            # Receive sensor data from the client
            # Wait for 20 values, add them to the buffer and then update the plot
            for _ in range(20):
                raw_data, addr = client_socket.recvfrom(4)
                value = struct.unpack('!f', raw_data)[0]

                # Add the new value to the buffer
                y_values.addElement(value)

            # Update sensor data plot
            current_y_values = y_values.getBuffer()
            x_values = np.arange(0, len(current_y_values), 1)
            line_audio.set_data(x_values, current_y_values)

            # Redraw the plot
            fig.canvas.draw()

            # Pause for a short interval to control the refresh rate
            plt.pause(0.001)

    except KeyboardInterrupt:
        # Close the socket when interrupted
        host_socket.close()


if __name__ == '__main__':
    receive_data()