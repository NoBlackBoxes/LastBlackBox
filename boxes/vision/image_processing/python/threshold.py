import os
import time
import cv2
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

try:
    while True:
        # Capture frame
        gray = camera.capture(lores=True, gray=True)

        # Apply binary thresholding
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Convert back to RGB so the output remains 3-channel
        display = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

        # Update display stream
        stream.display(display)

except KeyboardInterrupt:
    stream.stop()
    camera.stop()
