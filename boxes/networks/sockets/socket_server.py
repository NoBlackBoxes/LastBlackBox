import socket
import random
import time

host = '127.0.0.1'
port = 1234

def generate_random_data(size):
    """Generate a random string of bytes with the specified size."""
    return bytes([random.randint(0, 255) for _ in range(size)])

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))
s.listen()
print(f"Server listening on {host}:{port}")

while True:
    conn, addr = s.accept()
    print(f"Connected by {addr}")
    
    # Send random data to the client
    for _ in range(5):  # Send 5 random buffers of data
        buffer_size = random.randint(10, 100)  # Random buffer size between 10 and 100 bytes
        data = generate_random_data(buffer_size)
        conn.sendall(data)
        print(f"Sent {len(data)} bytes to the client")
        time.sleep(1)  # Simulate some delay between sending data
    
    print("Closing connection with the client.")
    conn.close()

# FIN
