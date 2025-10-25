# -*- coding: utf-8 -*-
"""
NB3 : Vision : Camera Class

@author: kampff
"""

# Imports
import time
import cv2
import numpy as np
from threading import Lock, Condition
from picamera2 import Picamera2, MappedArray
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
import NB3.Vision.output as Output

# Camera Class
class Camera:
    def __init__(self, width, height, lores_width=None, lores_height=None, index=0):
        self.width = width
        self.height = height
        self.lores_width = lores_width if lores_width else width
        self.lores_height = lores_height if lores_height else height
        self.index = index
        self.num_channels = 3
        self.handle = Picamera2()
        self.mutex = Lock()
        self.overlay = None

        # Configure camera (main and low resolution images)
        config = self.handle.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            lores={"size": (self.lores_width, self.lores_height), "format": "YUV420"},
        )
        self.handle.configure(config)
        self.encoder = MJPEGEncoder(bitrate=self._set_bitrate())
        self.output = Output.Output()
        self.handle.pre_callback = self._pre_callback

    def start(self):
        self.handle.start()
        time.sleep(0.1)
        self.handle.start_encoder(self.encoder, FileOutput(self.output))
        return

    def stop(self):
        self.handle.stop_encoder()
        self.handle.stop()
        return

    def capture(self, lores=False, gray=False):
        with self.mutex:
            if lores:
                if gray:
                    frame = self.handle.capture_array("lores")[:self.lores_height,:]
                else:
                    frame = self.handle.capture_array("lores")
                    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)
            else:
                if gray:
                    frame = self.handle.capture_array("main")
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    frame = self.handle.capture_array("main")
            return frame

    def mjpeg(self):
        with self.mutex:
            return self.output.get_frame()

    def save(self, filename):
        with self.mutex:
            frame = self.handle.capture_array("main")
            cv2.imwrite(filename, frame)
            return

    def display(self, frame, server, stream, jpeg=False):
        if not jpeg:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            _, encoded = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame = encoded.tobytes()
        server.update_stream(stream, frame)
        return

    def _pre_callback(self, request):
        if self.overlay:
            with MappedArray(request, "main") as m:
                self.overlay.draw(m.array)
        return

    def _set_bitrate(self):
        reference_complexity = 1920 * 1080 * 30
        actual_complexity = self.width * self.height * 30
        reference_bitrate = 40 * 1000000
        return int(reference_bitrate * actual_complexity / reference_complexity)
#FIN