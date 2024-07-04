# Capture an image and save it to a PNG file

import time
from picamera2 import Picamera2

# Open the Camera
camera = Picamera2()

# Start the Camera
camera.start()

# Wait two seconds
time.sleep(2)

# Save the latest image to a file
camera.capture_file("test.png")
