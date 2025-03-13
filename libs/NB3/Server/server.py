# -*- coding: utf-8 -*-
"""
NB3 : Server : Server Class

@author: kampff
"""

# Imports
import os
import re
import time
import serial
import socket
import netifaces
import threading

class Server:
    def __init__(self, root, port=1234, interface="wlan0", command_handler=None):
        self.root = os.path.abspath(root)                                                # Site root folder
        self.port = port                                                                 # Server port
        self.ip_address = netifaces.ifaddresses(interface)[netifaces.AF_INET][0]['addr'] # Get IP address for interface
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)           # Server socket
        self.command_handler = command_handler                                           # Command handler callback
        self.stream_names = self.extract_stream_names(os.path.join(root, "index.html"))  # Extract stream names
        self.stream_clients = {}
        for stream_name in self.stream_names:
            self.stream_clients[stream_name] = []
        self.running = False
        self.server_thread = None
    
    def start(self):
        self.running = True
        self.server_thread = threading.Thread(target=self._accept_clients, daemon=True)
        self.server_thread.start()

    def stop(self):
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.server_socket.close()
        for stream_name, clients in self.stream_clients.items():
            for client_socket in clients:
                try:
                    client_socket.shutdown(socket.SHUT_RDWR)
                    client_socket.close()
                except OSError:
                    pass
            clients.clear()

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
                if self.running:
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
                # - Leave client socket open -
            elif path.startswith("/command/"):
                command = path.split("/")[-1]
                self.handle_command(command)
                response = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nCommand Received"
                client_socket.sendall(response)
                client_socket.close()
            else:
                self.serve_file(client_socket, os.path.join(self.root, path[1:]))
                client_socket.close()
        except (BrokenPipeError, ConnectionResetError, OSError):
            print(f"NB3 Server: Client disconnected.")
        except Exception as e:
            print(f"NB3 Server client error: {e}")

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
        elif file_path.endswith(".svg"):
            content_type = "image/svg+xml"
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
        
        # Add client to this stream's client list
        self.stream_clients[stream_name].append(client_socket)

        # Send HTTP response header once
        try:
            headers = (
                "HTTP/1.1 200 OK\r\n"
                "Age: 0\r\n"
                "Cache-Control: no-cache, private\r\n"
                "Pragma: no-cache\r\n"
                "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
            )
            client_socket.sendall(headers.encode('utf-8'))
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.stream_clients[stream_name].remove(client_socket)
            client_socket.close()  # Close the socket if an error occurs

    def update_stream(self, stream_name, frame):
        if stream_name not in self.stream_names:
            return

        # Send encoded frame to all stream clients
        multipart_frame = (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(frame)}\r\n\r\n".encode()
            + frame
        )
        clients_to_remove = []
        for client_socket in self.stream_clients[stream_name]:
            try:
                client_socket.sendall(multipart_frame)
            except (BrokenPipeError, ConnectionResetError, OSError):
                clients_to_remove.append(client_socket)  # Mark for removal
        for client_socket in clients_to_remove:
            self.stream_clients[stream_name].remove(client_socket)
            client_socket.close()
    
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

    def handle_command(self, command):
        if self.command_handler:
            try:
                self.command_handler(command)
            except Exception as e:
                print(f"NB3 Server: Error in command handler: {e}")
        else:
            print(f"NB3 Server: No command handler set. Received: {command}")

# Utility
def get_wifi_interface():
    for interface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addrs:
            if "wlan" in interface or "wlp" in interface:
                return interface
    return None
