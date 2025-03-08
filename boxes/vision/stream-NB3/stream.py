# Streaming image (MJPEG) HTTP server

# Imports
import os
import time
import socket
import netifaces
import mimetypes
import numpy as np
import cv2
import NB3.Vision.camera as Camera

# Load page data (index.html)
path = 'index.html'
file = open(path,"r")
html = file.read()
file.close()

# Get Host (NB3) IP address of the WiFI interface
ip_address = netifaces.ifaddresses("wlan0")[netifaces.AF_INET][0]['addr']

# Set port
port = 1234

# Open socket
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((ip_address, port))
listen_socket.listen(1)

# Open camera, start, and wait for it to settle
camera = Camera.Camera(type='picamera2', device=0, width=640, height=480, format='RGB')
camera.start()
time.sleep(1.0)

# Clear screen and print status
os.system('clear')
print(f"\nConnect to http://{ip_address}:{port} to view camera stream")  
print(f" - (Ctrl-C to Quit)")

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

            # Continuously send (stream) of JPEG images
            try:
                while True:
                    # Capture latest image
                    frame = camera.latest()

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
            pass

        # Close client connection
        client_connection.close()
except:
    # Stop the camera
    camera.stop()

#FIN