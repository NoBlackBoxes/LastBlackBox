# -*- coding: utf-8 -*-
"""
NB3 : Server : Server Class

@author: kampff
"""

# Imports
import os
import cv2
import socket
import netifaces
import threading
import numpy as np
import time

class Server:
    def __init__(self, root, port=1234, interface="wlan0"):
        self.root = os.path.abspath(root)
        self.port = port
        self.ip_address = netifaces.ifaddresses(interface)[netifaces.AF_INET][0]['addr']
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.running = False
        self.server_thread = None
        self.stream_frames = {}
        self.stream_locks = {}
        self.stream_clients = {}

    def start(self):
        if self.server_thread and self.server_thread.is_alive():
            print("NB3 Server already running!")
            return
        self.running = True
        self.server_thread = threading.Thread(target=self._accept_clients, daemon=True)
        self.server_thread.start()

    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()

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
            request = client_socket.recv(1024).decode("utf-8")
            if not request:
                client_socket.close()
                return

            request_line = request.split("\r\n")[0]
            method, path, _ = request_line.split(" ")
            path = path.replace('%20', ' ').replace('%22', '"').replace('%27', "'")

            requested_path = os.path.normpath(os.path.join(self.root, path.lstrip('/')))
            if not requested_path.startswith(self.root):
                self.send_404(client_socket)
                client_socket.close()
                return

            if path == "/":
                self.serve_file(client_socket, os.path.join(self.root, "index.html"), "text/html")
                client_socket.close()
            elif path.startswith("/stream/"):
                stream_name = path.split("/")[-1].replace(".mjpg", "")
                self.serve_stream(client_socket, stream_name)
            else:
                self.serve_file(client_socket, requested_path)
                client_socket.close()

        except Exception as e:
            print(f"NB3 Server client error: {e}")
            client_socket.close()

    def serve_file(self, client_socket, file_path, content_type=None):
        if not os.path.exists(file_path):
            self.send_404(client_socket)
            return

        if content_type is None:
            content_type = "application/octet-stream"
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

        with open(file_path, "rb") as f:
            content = f.read()

        response = (
            f"HTTP/1.1 200 OK\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(content)}\r\n\r\n"
        ).encode("utf-8") + content

        client_socket.sendall(response)

    def serve_stream(self, client_socket, stream_name):
        """Continuously sends the latest frame to all clients in a dedicated thread per stream."""

        # Ensure stream is initialized properly
        if stream_name not in self.stream_frames:
            self.stream_frames[stream_name] = None
            self.stream_locks[stream_name] = threading.Lock()
            self.stream_clients[stream_name] = []  # Initialize client list
            threading.Thread(target=self._stream_loop, args=(stream_name,), daemon=True).start()  # Ensure it runs

        # Add the client to the list
        with self.stream_locks[stream_name]:
            self.stream_clients[stream_name].append(client_socket)

        # Send HTTP response header once
        try:
            client_socket.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
            )
        except (BrokenPipeError, ConnectionResetError, OSError):
            with self.stream_locks[stream_name]:
                self.stream_clients[stream_name].remove(client_socket)

    def _stream_loop(self, stream_name):
        """Main loop for continuously sending frames to all connected clients."""        
        while self.running:
            with self.stream_locks.get(stream_name, threading.Lock()):
                frame = self.stream_frames.get(stream_name, None)  # Avoid KeyError
                clients = list(self.stream_clients.get(stream_name, []))  # Avoid KeyError

            if frame is None or not clients:
                time.sleep(0.05)  # No frame or no clients, avoid busy looping
                continue

            multipart_frame = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                + frame
            )

            disconnected_clients = []
            for client in clients:
                try:
                    client.sendall(multipart_frame)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    disconnected_clients.append(client)  # Mark for removal

            # Remove disconnected clients safely
            with self.stream_locks[stream_name]:
                for client in disconnected_clients:
                    self.stream_clients[stream_name].remove(client)

            time.sleep(0.05)  # Controls frame rate!!!

    def update_stream(self, stream_name, frame):
        """Update the latest frame for a stream and ensure thread safety."""

        # Ensure the stream is available
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

        # Store the latest frame safely
        with self.stream_locks[stream_name]:
            self.stream_frames[stream_name] = frame

    def send_404(self, client_socket):
        response = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"
        client_socket.sendall(response)

# Utility
def get_wifi_interface():
    for interface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addrs:
            if "wlan" in interface or "wlp" in interface:
                return interface
    return None
