import time
import cv2
import numpy as np
from threading import Lock

#
# Video input (camera)
#
class camera:
    def __init__(self, type, device, width, height, format):
        self.width = width
        self.height = height
        self.format = format
        self.num_channels = None
        self.mutex = Lock()

        # Select image format
        if format == 'RGB' or 'BGR':
            self.num_channels = 3
        elif format == 'RGBA' or 'BGRA':
            self.num_channels = 4
        else:
            self.num_channels = 1

        # Create buffers
        self.current = np.zeros((self.height, self.width, self.num_channels), dtype=np.uint8)

    # Start camera
    def start(self):
        return

    # Stop camera
    def stop(self):
        return

    # Copy latest image data
    def latest(self):
        return

# FIN