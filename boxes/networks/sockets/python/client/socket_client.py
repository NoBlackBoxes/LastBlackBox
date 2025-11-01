# Socket Client
#  This program should run on your PC. It will connect to your NB3 and then receive data streamed
# over the socket.
import socket
import numpy as np

# Specify host IP address and port to use for the socket.
ip_address = '192.168.1.70'  # Use the IP of your NB3 (the server)
port = 1234  # Use the same port as specified in the socket_server

# Create a Socket that establish the server connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(60)  # Timeout after 60 seconds of inactivity
sock.connect((ip_address, port))
print(f"Connected to server at {ip_address}:{port}")

# The Socket Client Loop
samples_per_buffer = 16
try:
    while True:
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
        data = np.frombuffer(buffer, dtype=np.uint8) # Convert to a numpy array
        print(f"Received: {data}")
except socket.timeout:
    print("Socket timed out. No data received for 60 seconds.")
except Exception as e:
    print(f"Error: {e}")
finally:
    sock.close()

#FIN
