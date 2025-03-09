import os
import time
import cv2
import numpy as np
import NB3.Vision.camera as Camera
import NB3.Vision.stream as Stream
import NB3.Vision.overlay as Overlay

# Setup Camera
camera = Camera.Camera(width=1280, height=720, lores_width=640, lores_height=480)
camera.start()

# Add Overlay
overlay = Overlay.Overlay()
overlay.timestamp = True
camera.overlay = overlay

# Setup MJPEG stream
stream = Stream.Stream(camera=camera, port=1234, lores=True)
stream.start()

# Define HSV range for green color
lower_green = np.array([35, 90, 70])   # Lower bound (H, S, V)
upper_green = np.array([85, 255, 255]) # Upper bound (H, S, V)

try:
    while True:
        # Capture frame (RGB)
        rgb = camera.capture(lores=True, gray=False)

        # Convert to HSV
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # Threshold for GREEN
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Convert mask to RGB so the output remains 3-channel
        display = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the largest contour based on area
            largest_contour = max(contours, key=cv2.contourArea)

            # Draw the largest contour on the original frame
            cv2.drawContours(display, [largest_contour], -1, (0, 255, 0), 3)  # Green contour

        # Update display stream
        stream.display(display)

except KeyboardInterrupt:
    stream.stop()
    camera.stop()
