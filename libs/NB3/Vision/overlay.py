# -*- coding: utf-8 -*-
"""
NB3 : Vision : Overlay Class

@author: kampff
"""

# Imports
import cv2
import datetime
import numpy as np

# Overlay Class
class Overlay:
    def __init__(self):
        self.timestamp = False
        self.timestamp_color = (255, 255, 0)
        self.rectangle = None
        self.rectangle_color = (0, 255, 0)

    def draw(self, array):
        if self.timestamp:
            r, g, b = self.timestamp_color
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            cv2.putText(array, timestamp, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (b,g,r), 2, cv2.LINE_AA)
        if self.rectangle:
            x, y, w, h = self.rectangle
            r, g, b = self.rectangle_color
            cv2.rectangle(array, (int(x), int(y)), (int(x + w), int(y + h)), (b, g, r), 2)
        return
