# Capture an image, process it with numpy, and save it to a PNG file

import time
import numpy as np
from PIL import Image
from picamera2 import Picamera2

# Open the Camera
camera = Picamera2()

# Start the Camera
camera.start()

# Wait two seconds
time.sleep(2)

# Capture image to 3-D Numpy array [rows, cols, channels]
array = camera.capture_array("main")

# Process image (binary threshold)
binary = array > 127

# Convert Numpy array to image
image = Image.fromarray(binary)

# Save the processed image to a file
image.save('binary.png')
