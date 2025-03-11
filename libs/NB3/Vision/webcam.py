# -*- coding: utf-8 -*-
"""
NB3 : Vision : Webcam Class

@author: kampff
"""

# Imports
import time
import cv2
import numpy as np
from threading import Lock, Condition

# Webcam Class
class Webcam:
    def __init__(self, width, height, device=0):
        self.width = width
        self.height = height
        self.num_channels = 3
        self.device = device
        self.handle = None
        self.mutex = Lock()
        self.overlay = None

    def start(self):
        self.handle = cv2.VideoCapture(self.device)
        self.handle.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.handle.set(cv2.CAP_PROP_CONVERT_RGB, 0) # Disable automatic decoding (get raw MJPG buffer)
        self.handle.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.handle.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        time.sleep(0.1)
        return

    def status(self):
        #print(cv2.getBuildInformation())
        format_id = self.handle.get(cv2.CAP_PROP_FOURCC)
        print("Camera Format:", format_id)
        width = self.handle.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.handle.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera Resolution: {int(width)}x{int(height)}")

    def stop(self):
        self.handle.release()
        return

    def capture(self, gray=False):
        with self.mutex:
            ret, jpeg_buffer = self.handle.read() # Read raw MJPG frame
            if not ret or jpeg_buffer is None:
                print("Error: Failed to capture MJPG frame!")
                return None
            frame = cv2.imdecode(np.frombuffer(jpeg_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
            if gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame  # Return decoded frame

    def mjpeg(self):
        with self.mutex:
            ret, frame = self.handle.read()
            if not ret or frame is None:
                print("Error: Failed to capture MJPG frame!")
                return None
            return frame.tobytes()

    def save(self, filename):
        with self.mutex:
            ret, frame = self.handle.read()
            cv2.imwrite(filename, frame)
            return
#FIN