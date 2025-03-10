# -*- coding: utf-8 -*-
"""
NB3 : Vision : Output Class

@author: kampff
"""

# Imports
import io
import threading

# Output Class
class Output(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

    def get_frame(self):
        with self.condition:
            self.condition.wait()
            return self.frame
