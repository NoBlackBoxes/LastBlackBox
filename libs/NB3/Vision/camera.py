import time
import cv2
import numpy as np
from pprint import *
from threading import Lock
from picamera2 import Picamera2

#
# Video input (camera)
#
class Camera:
    def __init__(self, type, device, width, height, format):
        self.width = width
        self.height = height
        self.format = format
        self.num_channels = None
        self.mutex = Lock()
        self.handle = Picamera2()

        # Select image format and num_channels
        if format == 'RGB':
            self.num_channels = 3
            self.picam_format = 'BGR888'
        elif format == 'BGR':
            self.num_channels = 3
            self.picam_format = 'RGB888'
        elif format == 'RGBA':
            self.num_channels = 4
            self.picam_format = 'XBGR8888'
        elif format == 'BGRA':
            self.num_channels = 4
            self.picam_format = 'XRGB8888'
        else:
            self.num_channels = 1
            self.picam_format = 'YUV420'
        
        # Configure camera
        config = self.handle.create_preview_configuration(main={"size": (self.width, self.height),"format": self.picam_format})
        self.handle.configure(config)

        # Create buffer for current frame
        self.current = np.zeros((self.height, self.width, self.num_channels), dtype=np.uint8)

    # Report sensor modes
    def report_modes(self):
        pprint(self.handle.sensor_modes)
        return

    # Start camera
    def start(self):
        self.handle.start()
        time.sleep(0.1)  # Allow time for camera to warm up
        return

    # Stop camera
    def stop(self):
        self.handle.stop()
        return

    # Copy latest image data
    def latest(self):
        with self.mutex:
            frame = self.handle.capture_array()
            
            # Convert if necessary
            if self.format == 'BGR':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif self.format == 'BGRA':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
            elif self.format == 'GRAY':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Copy to current buffer
            np.copyto(self.current, frame)

        return self.current

    # Save current frame to file
    def save(self, filename):
        with self.mutex:
            cv2.imwrite(filename, self.current)
        return

# Example usage:
# cam = camera(type='picamera2', device=0, width=640, height=480, format='BGR')
# cam.start()
# frame = cam.latest()
# cam.stop()
