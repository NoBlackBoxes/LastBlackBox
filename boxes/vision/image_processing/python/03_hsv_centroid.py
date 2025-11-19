import os, pathlib, time
import cv2
import numpy as np
#import NB3.Vision.camera as Camera         # NB3 Camera
import NB3.Vision.webcam as Camera        # Webcam (PC)
import NB3.Server.server as Server

# Specify site root
repo_path = f"{pathlib.Path.home()}/NoBlackBoxes/LastBlackBox"
site_root = f"{repo_path}/boxes/vision/image_processing/python/sites/split"

# Setup Camera
camera = Camera.Camera(width=800, height=600, lores_width=640, lores_height=480)
camera.overlay.timestamp = True
camera.start()

# Start Server (for streaming)
interface = Server.get_wifi_interface()
server = Server.Server(root=site_root, interface=interface, autostart=True)

# Define HSV range for green color
lower_green = np.array([35, 90, 70])   # Lower bound (H, S, V)
upper_green = np.array([85, 255, 255]) # Upper bound (H, S, V)

try:
    while True:
        # Capture frame (RGB)
        rgb = camera.capture(mjpeg=False, lores=False, gray=False)

        # Convert to HSV
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # Threshold for GREEN
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        camera.overlay.clear()
        if contours:
            # Find the largest contour based on area
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            camera.overlay.add_rectangle(x, y, w, h)

        # Display raw and processed frames
        camera.display(rgb, server, "camera", overlay=True, jpeg=False)
        camera.display(mask, server, "display", overlay=True, jpeg=False, gray=True)

        # Delay
        time.sleep(0.033) # Limit to 30 FPS

except KeyboardInterrupt:
    server.stop()
    camera.stop()
