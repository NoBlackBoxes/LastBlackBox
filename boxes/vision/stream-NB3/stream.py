import os
import time
from NB3.Vision.camera import Camera
from NB3.Vision.stream import MJPEGStreamer

# Get user name
username = os.getlogin()

# Load external index.html
html_path = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/vision/stream-NB3/index.html"

# Setup Camera
camera = Camera(width=1280, height=720)
camera.start()

# Set rectangle overlay (x, y, width, height)
camera.set_rectangle_overlay(100, 100, 200, 150)
camera.clear_rectangle_overlay()

# Start MJPEG stream
streamer = MJPEGStreamer(camera=camera, port=1234, html_path=html_path)
streamer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    streamer.stop()
    camera.stop()
