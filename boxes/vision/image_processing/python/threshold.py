import os
import time
import cv2
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

try:
    print(f"    - \"Control + C\" to Quit -")
    while True:
        # Capture frame
        gray = camera.capture(lores=True, gray=True)

        # Apply binary thresholding
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Convert back to RGB so the output remains 3-channel
        display = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

        # Update streams
        frame = camera.mjpeg()
        server.update_stream("camera", frame)
        server.update_stream("display", display)

except KeyboardInterrupt:
    server.stop()
    camera.stop()
