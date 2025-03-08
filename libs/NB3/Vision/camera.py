import time
import cv2
import numpy as np
from threading import Lock
from picamera2 import Picamera2

class Camera:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.mutex = Lock()
        self.handle = Picamera2()
        self.num_channels = 3

        # Configure camera
        config = self.handle.create_video_configuration(main={"size": (self.width, self.height), "format": "RGB888"})
        self.handle.configure(config)

        # Create buffer for raw frames
        self.current = np.zeros((self.height, self.width, self.num_channels), dtype=np.uint8)

    def start(self):
        """Start the camera"""
        self.handle.start()
        time.sleep(0.1)  # Allow time for the camera to warm up

    def stop(self):
        """Stop the camera"""
        self.handle.stop()

    def latest(self):
        """Retrieve the latest frame as either raw or MJPEG-encoded data"""
        with self.mutex:
            frame = self.handle.capture_array()
            np.copyto(self.current, frame)
            return self.current

    def save(self, filename):
        """Save the latest frame"""
        with self.mutex:
            cv2.imwrite(filename, self.current)
#FIN