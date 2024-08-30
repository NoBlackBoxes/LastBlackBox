# Capture a PNG
import time
from picamera2 import Picamera2

picam2 = Picamera2()

picam2.start()
time.sleep(2)

picam2.capture_file("test.png")
