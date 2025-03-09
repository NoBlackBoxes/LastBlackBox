# -*- coding: utf-8 -*-
"""
NB3 : Vision : Stream Class

@author: kampff
"""

# Imports
import os
import io
import cv2
import time
import socket
import threading
import netifaces
import numpy as np

# Set default HTML page
default_html = b"""<html>
    <head>
    <title>NB3 Camera Streaming</title>
    <style>
        html, body {
            margin: 0;
            padding: 0;
            height: 100%;
            width: 100%;
            overflow: hidden;
            display: flex;
            flex-direction: row;
        }
        .container {
            width: 50vw;
            height: 100vh;
        }
        .container img {
            width: 100%;
            height: 100%;
            object-fit: fill;
        }
        @media (orientation: portrait) {
            body {
                flex-direction: column;
            }
            .container {
                width: 100vw;
                height: 50vh;
            }
        }
    </style>
</head>
<body>
    <div class="container"><img src="stream.mjpg" /></div>
    <div class="container"><img src="display.mjpg" /></div>
</body>
</html>"""

# Stream Class
class Stream:
    def __init__(self, camera, port=1234, html_path=None, lores=False):
        self.camera = camera
        self.port = port
        self.html = self.load_html(html_path)
        self.lores = lores
        self.listen_socket = None
        self.mutex = threading.Lock()    
        self.running = False

        # Shared frame buffers
        self.current_frame = None  # Raw MJPEG frame from the camera
        if lores:
            self.display_frame = np.zeros((camera.lores_height, camera.lores_width, camera.num_channels), dtype=np.uint8)
        else:
            self.display_frame = np.zeros((camera.height, camera.width, camera.num_channels), dtype=np.uint8)

        # Client lists
        self.stream_clients = []
        self.display_clients = []

    def load_html(self, path):
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        return default_html

    def start(self):
        ip_address = netifaces.ifaddresses("wlan0")[netifaces.AF_INET][0]['addr']
        self.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listen_socket.bind((ip_address, self.port))
        self.listen_socket.listen(5)
        self.running = True
        print(f"\n🌐 Camera stream running at http://{ip_address}:{self.port}")
        
        # Start accept and image streaming threads
        threading.Thread(target=self._accept_clients, daemon=True).start()
        threading.Thread(target=self._stream, daemon=True).start()
        threading.Thread(target=self._display, daemon=True).start()

    def display(self, frame):
        """Updates the display frame with processed data."""
        with self.mutex:
            np.copyto(self.display_frame, frame)

    def _accept_clients(self):
        """Accepts incoming client connections and adds them to the appropriate stream list."""
        while self.running:
            try:
                client_connection, _ = self.listen_socket.accept()
                threading.Thread(target=self._handle_client, args=(client_connection,), daemon=True).start()
            except OSError:
                break  # Exit loop when server stops

    def _handle_client(self, client_connection):
        """Handles HTTP requests and adds clients to the correct stream list."""
        try:
            request = client_connection.recv(1024).decode('utf-8')
            target = request.split(" ")[1] if len(request.split(" ")) > 1 else "/"
            if target == "/":
                response = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n" + self.html
                client_connection.sendall(response)
                client_connection.close()
            elif target == "/stream.mjpg":
                with self.mutex:
                    self.stream_clients.append(client_connection)
            elif target == "/display.mjpg":
                with self.mutex:
                    self.display_clients.append(client_connection)
            else:
                client_connection.close()

        except (BrokenPipeError, ConnectionResetError):
            print("❌ Client disconnected before fully connecting.")

        finally:
            if target not in ["/stream.mjpg", "/display.mjpg"]:
                client_connection.close()

    def _stream(self):
        """Continuously sends raw MJPEG frames from the camera to all clients."""
        while self.running:
            frame = self.camera.mjpeg()  # Get the MJPEG-encoded frame directly
            if frame is None:
                continue

            with self.mutex:
                if not self.stream_clients:
                    time.sleep(0.05)
                    continue

                for client in list(self.stream_clients):
                    try:
                        client.sendall(
                            b"HTTP/1.1 200 OK\r\n"
                            b"Cache-Control: no-cache\r\n"
                            b"Pragma: no-cache\r\n"
                            b"Content-Type: multipart/x-mixed-replace; boundary=FRAME\r\n\r\n"
                        )
                        client.sendall(b"--FRAME\r\n")
                        client.sendall(b"Content-Type: image/jpeg\r\n")
                        client.sendall(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        client.sendall(frame)

                    except (BrokenPipeError, ConnectionResetError, OSError):
                        self.stream_clients.remove(client)

            time.sleep(0.05)  # Controls frame rate

    def _display(self):
        """Encodes and sends MJPEG frames for the processed display stream."""
        while self.running:
            with self.mutex:
                frame = self.display_frame.copy()

            if frame is not None:
                _, encoded_frame = cv2.imencode('.JPEG', frame)

                with self.mutex:
                    if not self.display_clients:
                        time.sleep(0.05)
                        continue

                    for client in list(self.display_clients):  
                        try:
                            client.sendall(
                                b"HTTP/1.1 200 OK\r\n"
                                b"Cache-Control: no-cache\r\n"
                                b"Pragma: no-cache\r\n"
                                b"Content-Type: multipart/x-mixed-replace; boundary=FRAME\r\n\r\n"
                            )
                            client.sendall(b"--FRAME\r\n")
                            client.sendall(b"Content-Type: image/jpeg\r\n")
                            client.sendall(f"Content-Length: {len(encoded_frame)}\r\n\r\n".encode())
                            client.sendall(encoded_frame)

                        except (BrokenPipeError, ConnectionResetError, OSError):
                            self.display_clients.remove(client)

            time.sleep(0.05)  # Controls frame rate

    def stop(self):
        """Stops the stream and closes all connections."""
        self.running = False
        if self.listen_socket:
            self.listen_socket.close()

# Output Class
class Output(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

    def get_frame(self):
        with self.condition:
            self.condition.wait()
            return self.frame
