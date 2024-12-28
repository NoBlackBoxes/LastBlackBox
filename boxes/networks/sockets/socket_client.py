import socket

host = '172.20.62.243'  # Use the correct IP of your Android server
port = 1234  # Use the same port

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(60)  # Timeout after 60 seconds of inactivity
s.connect((host, port))
print(f"Connected to server at {host}:{port}")

try:
    while True:
        data = s.recv(1024)
        if not data:
            #print("No data received, waiting...")  # Log when no data is received
            continue  # Keep the connection alive
        print(f"Received: {data.decode('utf-8')}")
except socket.timeout:
    print("Socket timed out. No data received for 60 seconds.")
except Exception as e:
    print(f"Error: {e}")
finally:
    s.close()

