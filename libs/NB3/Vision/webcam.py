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
import NB3.Vision.output as Output

# Webcam (Camera) Class
class Camera:
    def __init__(self, width, height, lores_width=None, lores_height=None, index=0):
        self.width = width
        self.height = height
        self.lores_width = lores_width if lores_width else width
        self.lores_height = lores_height if lores_height else height
        self.index = index
        self.num_channels = 3
        self.handle = None
        self.mutex = Lock()
        self.overlay = None
        self.encoder = None
        self.output = Output.Output()

    def start(self):
        self.handle = cv2.VideoCapture(self.index)
        self.handle.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.handle.set(cv2.CAP_PROP_CONVERT_RGB, 0) # Disable automatic decoding (get raw MJPG buffer)
        self.handle.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.handle.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        time.sleep(0.1)
        return

    def stop(self):
        self.handle.release()
        return

    def capture(self, lores=False, gray=False):
        with self.mutex:
            ret, jpeg_buffer = self.handle.read() # Read raw MJPG frame
            if not ret or jpeg_buffer is None:
                print("Error: Failed to capture MJPEG frame!")
                return None
            frame = cv2.imdecode(np.frombuffer(jpeg_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
            if gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.overlay:
                self.overlay.draw(frame)
            return frame  # Return decoded frame

    def mjpeg(self):
        with self.mutex:
            ret, jpeg_buffer = self.handle.read()
            if self.overlay:
                frame = cv2.imdecode(np.frombuffer(jpeg_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.overlay.draw(frame)
                _, jpeg_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return jpeg_buffer.tobytes()

    def save(self, filename):
        with self.mutex:
            ret, frame = self.handle.read()
            cv2.imwrite(filename, frame)
            return
        
    def _set_bitrate(self):
        pass
        return

#FIN