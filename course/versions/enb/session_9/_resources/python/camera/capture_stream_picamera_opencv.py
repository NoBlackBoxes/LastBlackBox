import socket
import threading
import numpy as np
import cv2
from picamera2 import Picamera2
import time

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
def serve_template(client_socket):
    with open('fullscreen.html', 'r') as f:
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
    camera_config = camera.create_video_configuration(main={"size": (1280, 720)})
    camera.configure(camera_config)
    camera.start()
    start_time = time.time()
    try:
        while live.is_set():
            array = camera.capture_array("main")
            frame = process_frame(array)
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                break

            mjpeg_frame = (f"--jpgboundary\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n\r\n").encode() + jpeg.tobytes()

            client_socket.sendall(mjpeg_frame)
            end_time = time.time()
            #print(end_time-start_time)
            start_time = end_time
    finally:
        camera.stop()
        camera.close()
        client_socket.close()

def process_frame(frame):
    binary = (frame > 140) * 255
    return frame

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
            if 'GET /stream.mjpg' in request:
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