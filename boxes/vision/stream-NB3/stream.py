import os
import time
import NB3.Vision.camera as Camera
import NB3.Vision.stream as Stream

# Get user name
username = os.getlogin()

# Load external index.html
html_path = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/vision/stream-NB3/index.html"

# Setup Camera
camera = Camera.Camera(width=1280, height=720)
camera.start()

# Start MJPEG stream
stream = Stream.Stream(camera=camera, port=1234, html_path=html_path)
stream.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    stream.stop()
    camera.stop()
