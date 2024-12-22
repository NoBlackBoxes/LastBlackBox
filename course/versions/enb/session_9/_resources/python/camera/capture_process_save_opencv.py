# Capture an image, process it with numpy, and save it to a PNG file

import time
import numpy as np
import cv2
from picamera2 import Picamera2

# Open the Camera
camera = Picamera2()

# Start the Camera
camera.start()

# Wait two seconds
time.sleep(2)

for i in range(100):
    # Capture image to 3-D Numpy array [rows, cols, channels]
    array = camera.capture_array("main")

    # Process image (binary threshold)
    red = array[:,:,0]
    ret, binary = cv2.threshold(red, 127, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(image=binary, threshold1=100, threshold2=200) # Canny Edge Detection
    print(i)

# Save the processed image to a file
cv2.imwrite('edges.png', edges)
