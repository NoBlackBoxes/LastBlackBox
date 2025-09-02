# Simple HTTP server
import socket
import serial
import time

# Load home page data (index.html)
path = "index.html"
file = open(path,"r")
html = file.read()
file.close()

# Configure serial port
ser = serial.Serial()
ser.baudrate = 115200
ser.port = '/dev/ttyUSB0'

# Open serial port
ser.open()
time.sleep(1.00) # Wait for connection before sending any data

# Load favicon
favicon_path = "favicon.ico"
with open(favicon_path, "rb") as f:
    favicon = f.read()

# Set host (this computer) and port (1234)
HOST, PORT = '', 1234

# Open socket
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((HOST, PORT))
listen_socket.listen(1)
print(f'Serving HTTP on port {PORT} ...')

# Serve incoming connections
try:
    while True:
        # Listen for a connection
        client_connection, client_address = listen_socket.accept()

        # When a connection arrives, retrieve/decode HTTP request
        request_data = client_connection.recv(1024)
        request_text = request_data.decode('utf-8')

        # Parse request
        lines = request_text.split("\r\n")

        # Select first line
        line = lines[0]

        # Parse fields
        fields = line.split(" ")

        # Parse target
        if(len(fields) < 2):
            continue
        target = fields[1]
        print(fields)

        # Respond to target
        if(target == '/'): # The default target, serve 'index.html'
            header = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"
            response = bytes(header+html, 'utf-8')
            client_connection.sendall(response)

        elif(target == '/favicon.ico'):
            header = "HTTP/1.1 200 OK\r\nContent-Type: image/x-icon\r\n\r\n"
            response = header.encode() + favicon
            client_connection.sendall(response)

        elif(target.startswith('/control')):
            print(target)
            path, query = target.split("?", 1)
            if query.startswith("direction="):
                direction = query.split("=", 1)[1]
            header = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"
            response = bytes(header+html, 'utf-8')
            client_connection.sendall(response)
            if direction == "stop":
                ser.write(b'x')
                time.sleep(0.05)
                print("Stop")
            elif direction == "forward":
                ser.write(b'f')
                time.sleep(0.05)
                print("Forward")
            elif direction == "backward":
                ser.write(b'b')
                time.sleep(0.05)
                print("Backward")
            elif direction == "left":
                ser.write(b'l')
                time.sleep(0.05)
                print("Left")
            elif direction == "right":
                ser.write(b'r')
                time.sleep(0.05)
                print("Right")
        else:
            header = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"
            body = "404 Not Found"
            response = header.encode() + body.encode()
            client_connection.sendall(response)

        # Close client connection
        client_connection.close()

finally:
    # Close serial port
    ser.close()

#FIN