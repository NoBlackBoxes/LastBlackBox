import os
import time
import cv2
import numpy as np
#import NB3.Vision.camera as Camera
import NB3.Vision.webcam as Camera
import NB3.Vision.overlay as Overlay
import NB3.Server.server as Server

# Specify site root
username = os.getlogin()
root = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/vision/image_processing/python/site"

# Setup Camera
camera = Camera.Camera(width=800, height=600, lores_width=640, lores_height=480)
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

# Define HSV range for green color
lower_green = np.array([35, 90, 70])   # Lower bound (H, S, V)
upper_green = np.array([85, 255, 255]) # Upper bound (H, S, V)

try:
    print(f"    - \"Control + C\" to Quit -")
    while True:
        # Capture frame (RGB)
        rgb = camera.capture(lores=True, gray=False)

        # Convert to HSV
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # Threshold for GREEN
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the largest contour based on area
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            overlay.rectangle1 = (x, y, w, h)

        # Convert mask to RGB so the output remains 3-channel
        rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        _, encoded = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        display = encoded.tobytes()

        # Update streams
        frame = camera.mjpeg()
        server.update_stream("camera", frame)
        server.update_stream("display", display)
        time.sleep(0.0333) # (Optional) Slow down stream to 30 FPS

except KeyboardInterrupt:
    server.stop()
    camera.stop()
