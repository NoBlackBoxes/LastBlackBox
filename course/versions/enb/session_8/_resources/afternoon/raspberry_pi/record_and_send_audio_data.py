import pyaudio
import socket
import numpy as np


# Host PC IP address and port number
SERVER_IP = '192.168.137.1'
SERVER_PORT = 5005


CHUNK_SIZE = 4096           # Buffer size
FORMAT = pyaudio.paInt16    # Data type
CHANNELS = 2                # Number of channels
RATE = 22050                # Sample rate (Hz)

def send_audio():
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open the audio stream
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)

    # Create a socket and connect to the server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_IP, SERVER_PORT))

    try:
        while True:
            # Read audio data from the stream
            data = stream.read(CHUNK_SIZE)

            # Send the audio data over the network
            client_socket.sendall(data)
    except KeyboardInterrupt:
        # Close the stream and socket when interrupted
        stream.stop_stream()
        stream.close()
        client_socket.close()

if __name__ == '__main__':
    send_audio()