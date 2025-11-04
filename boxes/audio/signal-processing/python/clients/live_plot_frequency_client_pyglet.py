# Plot Live Spectrum (Socket Client: pyglet)
#  This program should run on your PC. It will connect to your NB3 and then receive data streamed
# over the socket and plot it in a real-time graph, with LOW latency, using OpenGL.
import socket
import numpy as np
import NB3.Plot.line as Line

# Specify host IP address and port to use for the socket.
ip_address = '192.168.1.70'  # Use the IP of your NB3 (the server)
port = 1234  # Use the same port as specified in the socket_server

# Open line plot
line = Line.Line(min=-0.1, max=1.1, num_samples=2400)
line.open()

# Create a Socket that establish the server connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(60)  # Timeout after 60 seconds of inactivity
sock.connect((ip_address, port))
print(f"Connected to server at {ip_address}:{port}")

# Define receive buffer function (guarantee receipt of complete buffers)
def recv_buffer(buffer_size):
    buffer = bytearray()
    while len(buffer) < buffer_size:
        chunk = sock.recv(buffer_size - len(buffer))
        if not chunk:
            raise ConnectionError("socket closed")
        buffer += chunk
    return buffer

# The Socket Client Loop
samples_per_buffer = 2400 * 4   # Samples * 4-bytes per Float32
while True:
    try:
        buffer = recv_buffer(samples_per_buffer)
        if not buffer: # If buffer empty?
            print("No data received, waiting...")  # Log when no data is received
            continue  # Keep the connection alive
        data = np.frombuffer(buffer, dtype=np.float32)
        line.plot(data)
    except KeyboardInterrupt:
        break

# Cleanup
line.close()
sock.close()

#FIN
