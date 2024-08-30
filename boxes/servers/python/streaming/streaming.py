# Streaming image  (MJPEG) HTTP server
import os
import socket
import mimetypes
import numpy as np
import cv2

# Load home page data (index.html)
path = 'index.html'
file = open(path,"r")
html = file.read()
file.close()

# Set host (this computer) and port (1234)
HOST, PORT = '', 1234

# Open socket
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((HOST, PORT))
listen_socket.listen(1)
print(f'Serving HTTP on port {PORT} ...')

# Get video capture object for camera 0
cap = cv2.VideoCapture(0)

# Serve incoming connections
try:
    while True:
        # Listen for a connection
        client_connection, client_address = listen_socket.accept()

        # When a connection arrives, retrieve/decode HTTP request
        request_data = client_connection.recv(1024)
        request_text = request_data.decode('utf-8')
        print(request_text)

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
        print(target)

        # Respond to target
        if(target == '/'): # The default target, serve 'index.html'
            http_response = bytes(html, 'utf-8')
            client_connection.sendall(http_response)
        elif(target == '/stream.mjpg'):
            # Send header
            client_connection.sendall(b'HTTP/1.1 200 OK\r\n')
            client_connection.sendall(bytes('Age: 0\r\n', 'utf-8'))
            client_connection.sendall(bytes('Cache-Control: no-cache, private\r\n', 'utf-8'))
            client_connection.sendall(bytes('Pragma: no-cache\r\n', 'utf-8'))
            client_connection.sendall(bytes('Content-Type: multipart/x-mixed-replace; boundary=FRAME\r\n', 'utf-8'))
            client_connection.sendall(bytes('\r\n', 'utf-8'))

            # Continously send (stream) JPEG images
            try:
                while True:
                    # Read most recent frame
                    ret, frame = cap.read()

                    # Encode as JPEG
                    _, frame = cv2.imencode('.JPEG', frame)

                    # Send frame data (with some header info)
                    client_connection.sendall(b'--FRAME\r\n')
                    client_connection.sendall(bytes('Content-Type: image/jpeg\r\n', 'utf-8'))
                    client_connection.sendall(bytes('Content-Length: {0}\r\n'.format(len(frame)), 'utf-8'))
                    client_connection.sendall(bytes('\r\n', 'utf-8'))
                    client_connection.sendall(frame) # Send encoded data
            except:
                pass
        else:
            print("huh? I mean 404!")

        # Close client connection
        client_connection.close()
except:
    # Release the camera caputre
    cap.release()

#FIN