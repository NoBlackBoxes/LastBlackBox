import os
import time
import cv2
import numpy as np
import NB3.Vision.camera as Camera
import NB3.Vision.overlay as Overlay
import NB3.Server.server as Server

# Get user name
username = os.getlogin()

# Specify site root
root = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/vision/image_processing/python/site"

# Setup Camera
camera = Camera.Camera(width=1280, height=720, lores_width=640, lores_height=480)
camera.start()

# Add Overlay
overlay = Overlay.Overlay()
overlay.timestamp = True
camera.overlay = overlay

# Start Server (for streaming)
interface = Server.get_wifi_interface()
server = Server.Server(root=root, interface=interface)
server.start()
server.status()

# Create buffer for previous frame
previous = np.zeros((480,640), dtype=np.uint8)

try:
    print(f"    - \"Control + C\" to Quit -")
    while True:
        # Capture frame
        gray = camera.capture(lores=True, gray=True)

        # Compute absolute difference
        absdiff = cv2.absdiff(gray, previous)

        # Convert back to RGB so the output remains 3-channel
        display = cv2.cvtColor(absdiff, cv2.COLOR_GRAY2RGB)

        # Update streams
        frame = camera.mjpeg()
        server.update_stream("camera", frame, encoded=True)
        server.update_stream("display", display, encoded=False)
        time.sleep(0.0333) # (Optional) Slow down stream to 30 FPS

        # Store new previous
        previous = gray

except KeyboardInterrupt:
    server.stop()
    camera.stop()
