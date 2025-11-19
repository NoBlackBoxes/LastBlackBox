import os, pathlib, time
import cv2
import numpy as np
#import NB3.Vision.camera as Camera         # NB3 Camera
import NB3.Vision.webcam as Camera        # Webcam (PC)
import NB3.Vision.overlay as Overlay
import NB3.Server.server as Server

# Specify site root
repo_path = f"{pathlib.Path.home()}/NoBlackBoxes/LastBlackBox"
site_root = f"{repo_path}/boxes/vision/image_processing/python/sites/split"

# Setup Camera
camera = Camera.Camera(width=800, height=600, lores_width=640, lores_height=480)
camera.start()

# Add Overlay
overlay = Overlay.Overlay()
overlay.timestamp = True
camera.overlay = overlay

# Start Server (for streaming)
interface = Server.get_wifi_interface()
server = Server.Server(root=site_root, interface=interface)
server.start()
server.status()

# Create buffer for previous frame
previous = np.zeros((480,640), dtype=np.uint8)

try:
    while True:
        # Capture frame
        gray = camera.capture(lores=True, gray=True)

        # Compute absolute difference
        absdiff = cv2.absdiff(gray, previous)

        # Convert back to RGB so the output remains 3-channel
        rgb = cv2.cvtColor(absdiff, cv2.COLOR_GRAY2RGB)
        _, encoded = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        display = encoded.tobytes()

        # Update streams
        frame = camera.capture(mjpeg=True)
        server.update_stream("camera", frame)
        server.update_stream("display", display)
        time.sleep(0.0333) # (Optional) Slow down stream to 30 FPS

        # Store new previous
        previous = gray

except KeyboardInterrupt:
    server.stop()
    camera.stop()
