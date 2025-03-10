# -*- coding: utf-8 -*-
"""
NB3 : Server : Server Class

@author: kampff
"""

# Imports
import os
import re
import cv2
import time
import serial
import socket
import netifaces
import threading
import numpy as np

class Server:
    def __init__(self, root, port=1234, interface="wlan0", serial_device=None):
        self.root = os.path.abspath(root)   # Site root folder
        self.port = port                    # Server port
        self.ip_address = netifaces.ifaddresses(interface)[netifaces.AF_INET][0]['addr'] # Get IP address
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)           # Create socket
        self.stream_names = self.extract_stream_names(os.path.join(root, "index.html"))  # Extract stream names
        self.stream_frames = {}
        self.stream_locks = {}              
        self.stream_conditions = {}
        for stream_name in self.stream_names:
            self.stream_frames[stream_name] = None
            self.stream_locks[stream_name] = threading.Lock()
            self.stream_conditions[stream_name] = threading.Condition(self.stream_locks[stream_name])
        self.running = False
        self.server_thread = None
        self.serial_device = None
        if serial_device:
            self.serial_device = serial.Serial()
            self.serial_device.baudrate = 115200
            self.serial_device.port = serial_device

    def start(self):
        if self.serial_device:
            self.serial_device.open()
            time.sleep(1.0)
        self.running = True
        self.server_thread = threading.Thread(target=self._accept_clients, daemon=True)
        self.server_thread.start()

    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if self.serial_device:
            self.serial_device.close()

    def status(self):
        if self.running:
            print(f"\n🌐 NB3 Server running at http://{self.ip_address}:{self.port}")
        else:
            print("\n🌐 NB3 Server not running")

    def _accept_clients(self):
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.ip_address, self.port))
        self.server_socket.listen(5)
        while self.running:
            try:
                client_socket, _ = self.server_socket.accept()
                threading.Thread(target=self._handle_client, args=(client_socket,), daemon=True).start()
            except Exception as e:
                print(f"NB3 Server accept error: {e}")

    def _handle_client(self, client_socket):
        try:
            # Receive client request
            request = client_socket.recv(1024).decode("utf-8")
            if not request:
                client_socket.close()
                return

            # Parse request
            request_line = request.split("\r\n")[0]
            method, path, _ = request_line.split(" ")
            path = path.replace('%20', ' ').replace('%22', '"').replace('%27', "'") # Replace URL codes

            # Serve requested URL path
            if path == "/":
                self.serve_file(client_socket, os.path.join(self.root, "index.html"))
                client_socket.close()
            elif path.startswith("/stream/"):
                stream_name = path.split("/")[-1].replace(".mjpg", "")
                self.serve_stream(client_socket, stream_name)
            elif path.startswith("/command/"):
                command = path.split("/")[-1]
                self.process_command(command)
                response = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nCommand Received"
                client_socket.sendall(response)
                client_socket.close()
            else:
                self.serve_file(client_socket, os.path.join(self.root, path))
                client_socket.close()

        except Exception as e:
            print(f"NB3 Server client error: {e}")
            client_socket.close()

    def serve_file(self, client_socket, file_path):
        if not os.path.exists(file_path):
            self.send_404(client_socket)
            return

        # Determine content type
        if file_path.endswith(".html"):
            content_type = "text/html"
        elif file_path.endswith(".css"):
            content_type = "text/css"
        elif file_path.endswith(".js"):
            content_type = "application/javascript"
        elif file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
            content_type = "image/jpeg"
        elif file_path.endswith(".png"):
            content_type = "image/png"
        else:
            content_type = "application/octet-stream"

        # Load content
        with open(file_path, "rb") as f:
            content = f.read()

        # Send response
        response = (
            f"HTTP/1.1 200 OK\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(content)}\r\n\r\n"
        ).encode("utf-8") + content
        client_socket.sendall(response)

    def serve_stream(self, client_socket, stream_name):
        if stream_name not in self.stream_names:
            print(f"NB3 Server: Invalid stream request: {stream_name}")
            self.send_404(client_socket)
            client_socket.close()
            return

        # Spawn a dedicated thread for this client
        threading.Thread(target=self._client_stream_loop, args=(client_socket, stream_name), daemon=True).start()

        # Send HTTP response header once
        try:
            client_socket.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
            )
        except (BrokenPipeError, ConnectionResetError, OSError):
            client_socket.close()  # Close the socket if an error occurs

    def _client_stream_loop(self, client_socket, stream_name):
        """Thread that waits for new frames and sends them to the client."""
        while self.running:
            with self.stream_conditions[stream_name]:  # Wait for new frames
                self.stream_conditions[stream_name].wait()  # Block until update_stream() signals

                frame = self.stream_frames[stream_name]
                if frame is None:
                    continue  # Skip if there's no frame yet

                multipart_frame = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                    + frame
                )

                try:
                    client_socket.sendall(multipart_frame)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break  # Exit thread if client disconnects
        client_socket.close()

    def update_stream(self, stream_name, frame):
        if stream_name not in self.stream_locks:
            return

        # Encode frame if it's a numpy array
        if isinstance(frame, np.ndarray):
            success, encoded_frame = cv2.imencode('.jpg', frame)
            if success:
                frame = encoded_frame.tobytes()
            else:
                print(f"Encoding failed for '{stream_name}'.")
                return

        # Store the latest frame safely and notify waiting clients
        with self.stream_conditions[stream_name]:
            self.stream_frames[stream_name] = frame
            self.stream_conditions[stream_name].notify_all()  # Wake up client threads

    def send_404(self, client_socket):
        response = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"
        client_socket.sendall(response)

    def extract_stream_names(self, index_path):
        if not os.path.exists(index_path):
            print("NB3 Server Error: index.html not found.")
            return []
        with open(index_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        stream_pattern = re.findall(r'/stream/([\w-]+)\.mjpg', html_content)        
        return list(set(stream_pattern))

    def process_command(self, command):
        if self.serial_device.is_open:
            if command == "forward_on":
                self.serial_device.write(b'f\n')  # Send 'f' to move forward
            elif command == "forward_off":
                self.serial_device.write(b'x\n')  # Stop
            elif command == "backward_on":
                self.serial_device.write(b'b\n')  # Send 'b' to move backward
            elif command == "backward_off":
                self.serial_device.write(b'x\n')  # Stop
            elif command == "left_on":
                self.serial_device.write(b'l\n')  # Turn left
            elif command == "left_off":
                self.serial_device.write(b'x\n')  # Stop
            elif command == "right_on":
                self.serial_device.write(b'r\n')  # Turn right
            elif command == "right_off":
                self.serial_device.write(b'x\n')  # Stop
# Utility
def get_wifi_interface():
    for interface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addrs:
            if "wlan" in interface or "wlp" in interface:
                return interface
    return None
