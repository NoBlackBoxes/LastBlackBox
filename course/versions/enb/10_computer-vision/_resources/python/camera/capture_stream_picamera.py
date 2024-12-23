import socket
import io
import threading
from threading import Condition
import numpy as np
import cv2
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

# To Do: Add a queue to allow mutliple users to access the same camera feed

# Configure hosting address and port
HOST = ''
PORT = 1234

# Thread event indicating that the server is "live"
live = threading.Event()
live.set()

# -----------------------------------------------------------------------------
# Helper Classes
# -----------------------------------------------------------------------------
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

def serve_template(client_socket):
    with open('stream.html', 'r') as f:
        html_content = f.read()
    encoded_content = html_content.encode()
    
    response = (f"HTTP/1.1 200 OK\r\n"
               f"Content-Type: text/html; charset=UTF-8\r\n"
               f"Content-Length: {len(encoded_content)}\r\n\r\n").encode() + encoded_content
    
    client_socket.sendall(response)
    client_socket.close()

def mjpeg_stream(client_socket):
    headers = f"HTTP/1.1 200 OK\r\n" \
              f"Content-Type: multipart/x-mixed-replace; boundary=--jpgboundary\r\n\r\n"
    client_socket.sendall(headers.encode())

    camera = Picamera2()
    camera_config = camera.create_video_configuration(main={"size": (640, 480)})
    camera.configure(camera_config)
    output = StreamingOutput()
    camera.start_recording(MJPEGEncoder(), FileOutput(output))
    
    try:
        while live.is_set():
            with output.condition:
                output.condition.wait()
                jpeg = output.frame
            mjpeg_frame = (f"--jpgboundary\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n\r\n").encode() + jpeg
            client_socket.sendall(mjpeg_frame)
    finally:
        camera.stop_recording()
        camera.stop()
        client_socket.close()

def process_frame(frame):
    binary = (frame > 40) * 255
    return binary

# -----------------------------------------------------------------------------

# Serve frames
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Set the SO_REUSEADDR option
    s.bind((HOST, PORT))
    s.listen()
    print(f"Listening on {HOST}:{PORT}")
    
    active_threads = []
    try:
        while live.is_set():
            conn, addr = s.accept()
            request = conn.recv(1024).decode('utf-8')
            print("connection")
            if 'GET /stream.mjpg' in request:
                print("mjpeg")
                t = threading.Thread(target=mjpeg_stream, args=(conn,))
                active_threads.append(t)
                t.start()
            else:
                t = threading.Thread(target=serve_template, args=(conn,))
                active_threads.append(t)
                t.start()

    except KeyboardInterrupt:
        live.clear()
        print("\nShutting down server...")
        s.close()

    # Ensure all threads stop gracefully
    for t in active_threads:
        t.join()

#FIN