# Imports
import os
import io
import cv2
import time
import serial
import socket
import threading
import netifaces
import numpy as np
import NB3.Vision.camera as Camera

# Server Class
class Server:
    def __init__(self, camera, port=1234, html_path=None):
        self.camera = camera
        self.port = port
        self.html = self.load_html(html_path)
        self.listen_socket = None
        self.mutex = threading.Lock()    
        self.running = False

        # Client lists
        self.stream_clients = []

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

    def _accept_clients(self):
        """Accepts incoming client connections and adds them to the appropriate stream list."""
        while self.running:
            try:
                client_connection, _ = self.listen_socket.accept()
                threading.Thread(target=self._handle_client, args=(client_connection,), daemon=True).start()
            except OSError:
                break  # Exit loop when server stops

    def _handle_client(self, client_connection):
        """Handles HTTP requests for HTML, stream, and commands."""
        try:
            request = client_connection.recv(1024).decode('utf-8')
            target = request.split(" ")[1] if len(request.split(" ")) > 1 else "/"

            if target == "/":
                response = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n" + self.html
                client_connection.sendall(response)
                client_connection.close()

            elif target.startswith("/command/"):
                # Extract command from URL
                command = target.split("/")[-1]
                self.process_command(command)
                response = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nCommand Received"
                client_connection.sendall(response)
                client_connection.close()

            elif target == "/stream.mjpg":
                with self.mutex:
                    self.stream_clients.append(client_connection)

            else:
                client_connection.close()

        except (BrokenPipeError, ConnectionResetError):
            print("❌ Client disconnected before fully connecting.")

        finally:
            if target not in ["/stream.mjpg"]:
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

    def process_command(self, command):
        """Processes control commands sent from the web interface."""
        if ser.is_open:
            if command == "forward_on":
                ser.write(b'f\n')  # Send 'F' to move forward
            elif command == "forward_off":
                ser.write(b'x\n')  # Stop
            elif command == "backward_on":
                ser.write(b'b\n')  # Send 'B' to move backward
            elif command == "backward_off":
                ser.write(b'x\n')  # Stop
            elif command == "left_on":
                ser.write(b'l\n')  # Turn left
            elif command == "left_off":
                ser.write(b'x\n')  # Stop
            elif command == "right_on":
                ser.write(b'r\n')  # Turn right
            elif command == "right_off":
                ser.write(b'x\n')  # Stop
            print(f"Sent command: {command}")

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

# -----------------
# Drone Application
# -----------------

# Get user name
username = os.getlogin()

# Load external index.html
html_path = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/vision/drone-NB3/index.html"

# Setup Camera
camera = Camera.Camera(width=1280, height=720)
camera.start()

# Start Server
stream = Server(camera=camera, port=1234, html_path=html_path)
stream.start()

# Configure serial port
ser = serial.Serial()
ser.baudrate = 115200
ser.port = '/dev/ttyUSB0'

# Open serial port
ser.open()
time.sleep(2.00) # Wait for connection before sending any data

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    stream.stop()
    camera.stop()
    ser.close()
