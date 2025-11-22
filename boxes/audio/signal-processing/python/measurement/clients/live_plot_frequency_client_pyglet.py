# Plot Live Spectrum (Socket Client: pyglet)
#  This program should run on your PC. It will connect to your NB3 and then receive data streamed
# over the socket and plot it in a real-time graph, with LOW latency, using OpenGL.
import socket
import numpy as np
import NB3.Plot.line as Line

# Specify host IP address and port to use for the socket.
ip_address = '10.187.158.92'  # Use the IP of your NB3 (the server)
port = 1234  # Use the same port as specified in the socket_server

# Open line plot
line = Line.Line(min=-0.1, max=1.1, num_samples=495, show_cursor=False, show_label=True)
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
samples_per_buffer = 496 # Samples per buffer (FFT spectrum)
bytes_per_buffer = samples_per_buffer * 4 # 4-bytes per Float32
try:
    while True:
        try:
            buffer = recv_buffer(bytes_per_buffer)
            if not buffer: # If buffer empty?
                print("No data received, waiting...")  # Log when no data is received
                continue  # Keep the connection alive
            data = np.frombuffer(buffer, dtype=np.float32)

            # Find peak frequency (in range 50 Hz to 5 kHz)
            peak_freq = (np.argmax(data) * 10) + 50

            # Place a label on the line plot at the peak frequency
            line.axes.label_position = peak_freq / (samples_per_buffer * 10) # Scale to range 0.0 to 1.0 for plot window
            line.axes.label_text = f"{peak_freq} Hz"

            # Plot the data
            line.plot(data)

        except KeyboardInterrupt:
            break
finally:
    # Cleanup
    line.close()
    sock.close()

#FIN
