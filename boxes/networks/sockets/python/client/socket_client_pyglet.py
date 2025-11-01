# Socket Client (pyglet)
#  This program should run on your PC. It will connect to your NB3 and then receive data streamed
# over the socket and plot it in a real-time graph, with LOW latency, using OpenGL.
import socket
import numpy as np
import NB3.Plot.line as Line

# Specify host IP address and port to use for the socket.
ip_address = '192.168.1.70'  # Use the IP of your NB3 (the server)
port = 1234  # Use the same port as specified in the socket_server

# Open line plot
line = Line.Line(min=0, max=255, num_samples=1600)
line.open()

# Create a Socket that establish the server connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(60)  # Timeout after 60 seconds of inactivity
sock.connect((ip_address, port))
print(f"Connected to server at {ip_address}:{port}")

# The Socket Client Loop
samples_per_buffer = 16
while True:
    try:
        # Receive N bytes of data in each socket buffer
        buffer = bytearray()
        while len(buffer) < samples_per_buffer:
            chunk = sock.recv(samples_per_buffer - len(buffer))
            if not chunk:
                raise ConnectionError("socket closed")
            buffer += chunk
        if not buffer: # If buffer empty?
            print("No data received, waiting...")  # Log when no data is received
            continue  # Keep the connection alive
        data = np.frombuffer(buffer, dtype=np.uint8)
        line.plot(data)
    except KeyboardInterrupt:
        break

# Cleanup
line.close()
sock.close()

#FIN
