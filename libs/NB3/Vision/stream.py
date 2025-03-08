import socket
import threading
import netifaces
import os
import logging
import time
from NB3.Vision.camera import Camera

class MJPEGStreamer:
    def __init__(self, camera, port=1234, html_path=None):
        self.camera = camera
        self.port = port
        self.html = self.load_html(html_path)
        self.listen_socket = None
        self.running = False

    def load_html(self, path):
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                html = f.read()
        else:
            html = """<html><head><title>NB3 Camera Streaming</title></head><body><h1>Hi from NB3!</h1><img src="stream.mjpg" width="1920" height="1080" /></body></html>"""
        return html

    def start(self):
        ip_address = netifaces.ifaddresses("wlan0")[netifaces.AF_INET][0]['addr']
        self.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listen_socket.bind((ip_address, self.port))
        self.listen_socket.listen(5)
        self.running = True
        print(f"🌐 Camera stream running at http://{ip_address}:{self.port}")
        threading.Thread(target=self._accept_clients, daemon=True).start()

    def _accept_clients(self):
        while self.running:
            client_connection, _ = self.listen_socket.accept()
            threading.Thread(target=self._handle_client, args=(client_connection,), daemon=True).start()

    def _handle_client(self, client_connection):
        try:
            request = client_connection.recv(1024).decode('utf-8')
            target = request.split(" ")[1] if len(request.split(" ")) > 1 else "/"

            if target == "/" and self.html:
                response = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n" + self.html
                client_connection.sendall(response)

            elif target == "/stream.mjpg":
                headers = (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Cache-Control: no-cache\r\n"
                    b"Pragma: no-cache\r\n"
                    b"Content-Type: multipart/x-mixed-replace; boundary=FRAME\r\n\r\n"
                )
                client_connection.sendall(headers)

                while self.running:
                    frame = self.camera.latest_mjpeg()
                    if frame:
                        client_connection.sendall(b"--FRAME\r\n")
                        client_connection.sendall(b"Content-Type: image/jpeg\r\n")
                        client_connection.sendall(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        client_connection.sendall(frame)
                    time.sleep(0.05)

        except (BrokenPipeError, ConnectionResetError):
            logging.info("Client disconnected")
        finally:
            client_connection.close()

    def stop(self):
        self.running = False
        if self.listen_socket:
            self.listen_socket.close()
        self.camera.stop()
