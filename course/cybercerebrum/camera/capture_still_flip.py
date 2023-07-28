# Capture a PNG
import time
import libcamera
from picamera2 import Picamera2

picam2 = Picamera2()

camera_config = picam2.create_preview_configuration()
camera_config["transform"] = libcamera.Transform(hflip=0, vflip=1)
picam2.configure(camera_config)

picam2.start()
time.sleep(2)

picam2.capture_file("test.png")
