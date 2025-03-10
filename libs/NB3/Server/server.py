# -*- coding: utf-8 -*-
"""
NB3 : Server : Server Class

@author: kampff
"""

# Imports
import os
import io
import time
import socket
import netifaces
import threading

class Server:
    def __init__(self, root, port=1234, interface="wlan0"):
        self.root = os.path.abspath(root)  # Ensure root is an absolute path
        self.port = port
        self.ip_address = netifaces.ifaddresses(interface)[netifaces.AF_INET][0]['addr']
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.mutex = threading.Lock()    
        self.running = False
        self.server_thread = None
        self.stream_frames = {}  # Stores latest frame per stream
        self.stream_locks = {}   # Locks for stream synchronization

    def start(self):
        """Start the server in a background thread."""
        if self.server_thread and self.server_thread.is_alive():
            print("NB3 Server is already running!")
            return
        self.running = True
        self.server_thread = threading.Thread(target=self._accept_clients, daemon=True)
        self.server_thread.start()

    def stop(self):
        """Stop the server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()

    def _accept_clients(self):
        """Main loop for accepting client connections."""
        self.server_socket.bind((self.ip_address, self.port))
        self.server_socket.listen(5)
        print(f"\n🌐 NB3 Server running at http://{self.ip_address}:{self.port}")
        while self.running:
            try:
                client_socket, client_address = self.server_socket.accept()
                threading.Thread(target=self.handle_client, args=(client_socket,), daemon=True).start()
            except Exception as e:
                print(f"Error: {e}")

    def handle_client(self, client_socket):
        """Handle incoming client requests."""
        try:
            request = client_socket.recv(1024).decode("utf-8")
            if not request:
                client_socket.close()
                return
            # Parse Request
            request_line = request.split("\r\n")[0]
            method, path, _ = request_line.split(" ")
            path = path.replace('%20', ' ').replace('%22', '"').replace('%27', "'") # Remove common URL-encoded chars

            # Prevent path traversal attack
            requested_path = os.path.normpath(os.path.join(self.root, path.lstrip('/')))
            if not requested_path.startswith(self.root):
                self.send_404(client_socket)
                return

            # Serve request
            if path == "/":
                self.serve_file(client_socket, os.path.join(self.root, "index.html"), "text/html")
            elif path.startswith("/stream/"):
                stream_name = path.split("/")[-1].replace(".jpg", "")
                self.handle_stream(client_socket, stream_name)
            else:
                self.serve_file(client_socket, requested_path)  # Serve file from root

        except Exception as e:
            print(f"NB3 Server: Client error: {e}")
        finally:
            client_socket.close()

    def serve_file(self, client_socket, file_path, content_type=None):
        """Serve static files (HTML, CSS, JS, images)."""
        if not os.path.exists(file_path):
            self.send_404(client_socket)
            return

        if content_type is None:
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

        with open(file_path, "rb") as f:
            content = f.read()

        response = (
            f"HTTP/1.1 200 OK\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(content)}\r\n\r\n"
        ).encode("utf-8") + content
        
        client_socket.sendall(response)

    def handle_stream(self, client_socket, stream_name):
        """Handle MJPEG streaming requests dynamically."""
        if stream_name not in self.stream_locks:
            self.stream_locks[stream_name] = threading.Condition()
            self.stream_frames[stream_name] = None  # No frame yet

        client_socket.sendall(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
        )

        try:
            while self.running:
                with self.stream_locks[stream_name]:
                    self.stream_locks[stream_name].wait()  # Wait for new frame
                    frame = self.stream_frames[stream_name]

                if frame:
                    client_socket.sendall(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" +
                        frame + b"\r\n"
                    )
        except (BrokenPipeError, ConnectionResetError):
            print(f"Client disconnected from {stream_name}")
        finally:
            client_socket.close()

    def update_stream(self, name, frame):
        """Push a new frame for a stream and notify waiting clients."""
        if name not in self.stream_locks:
            self.stream_locks[name] = threading.Condition()
            self.stream_frames[name] = None  # Initialize empty

        with self.stream_locks[name]:
            self.stream_frames[name] = frame
            self.stream_locks[name].notify_all()

    def send_404(self, client_socket):
        """Send a 404 Not Found response."""
        response = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"
        client_socket.sendall(response)

# -----------------
# Utility Functions
# -----------------
def get_wifi_interface():
    """Detects the Wi-Fi interface name dynamically."""
    for interface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addrs:
            if "wlan" in interface or "wlp" in interface:  # Common Wi-Fi interface naming
                return interface
    return None  # No Wi-Fi interface found