import io
import time
import cv2
import numpy as np
from threading import Lock, Condition
from picamera2 import Picamera2, MappedArray
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

    def get_frame(self):
        with self.condition:
            self.condition.wait()
            return self.frame

class Camera:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.mutex = Lock()
        self.handle = Picamera2()
        self.num_channels = 3
        self.overlay_enabled = True  # Overlay toggle
        self.overlay_rect = None     # (x,y,w,h) rectangle overlay

        config = self.handle.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            lores={"size": (self.width, self.height), "format": "YUV420"},
        )
        self.handle.configure(config)

        self.current = np.zeros((self.height, self.width, self.num_channels), dtype=np.uint8)
        self.encoder = MJPEGEncoder()
        self.output = StreamingOutput()
        self.handle.pre_callback = self.apply_overlay

    def start(self):
        self.handle.start()
        time.sleep(0.1)
        self.handle.start_encoder(self.encoder, FileOutput(self.output))

    def stop(self):
        self.handle.stop_encoder()
        self.handle.stop()

    def latest(self):
        with self.mutex:
            frame = self.handle.capture_array()
            np.copyto(self.current, frame)
            return self.current

    def latest_mjpeg(self):
        with self.mutex:
            return self.output.get_frame()

    def apply_overlay(self, request):
        if self.overlay_enabled:
            with MappedArray(request, "main") as m:
                # Add timestamp overlay
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(m.array, timestamp, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255,0,0), 2, cv2.LINE_AA)
                
                # Add rectangle overlay if set
                if self.overlay_rect:
                    x, y, w, h = self.overlay_rect
                    cv2.rectangle(m.array, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    def enable_overlay(self, enabled=True):
        self.overlay_enabled = enabled

    def set_rectangle_overlay(self, x, y, w, h):
        self.overlay_rect = (x, y, w, h)

    def clear_rectangle_overlay(self):
        self.overlay_rect = None
