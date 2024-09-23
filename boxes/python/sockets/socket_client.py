import socket

host = '127.0.0.1'
port = 1234

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))
print(f"Connected to server at {host}:{port}")
    
while True:
    data = s.recv(1024)  # Receive data from the server
    if not data:
        break
    print(f"Received {len(data)} bytes: {data}")

# FIN