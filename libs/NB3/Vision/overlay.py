# -*- coding: utf-8 -*-
"""
NB3 : Vision : Overlay Class

@author: kampff
"""

# Imports
import cv2
import datetime
import numpy as np

# High-contrast 16-color palette for OpenCV (BGR order)
color_palette_16 = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 128, 255),    # Sky Blue
    (0, 255, 128),    # Spring Green
    (255, 0, 128),    # Pink
    (128, 255, 0),    # Lime Yellow
    (255, 128, 128),  # Light Red
    (128, 128, 255),  # Light Blue
    (128, 255, 255),  # Light Cyan
    (255, 255, 128),  # Pale Yellow
]

# Overlay Class
class Overlay:
    def __init__(self):
        self.palette = color_palette_16
        self.timestamp = False
        self.timestamp_color = color_palette_16[3] # Yellow
        self.rectangles = []

    def add_rectangle(self, x, y, w, h):
        self.rectangles.append((x, y, w, h))

    def clear(self):
        self.rectangles = []

    def draw(self, array):
        if self.timestamp:
            r, g, b = self.timestamp_color
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            cv2.putText(array, timestamp, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (b,g,r), 2, cv2.LINE_AA)
        for i, rectangle in enumerate(self.rectangles):
            x, y, w, h = rectangle
            r, g, b = self.palette[i % 16]
            cv2.rectangle(array, (int(x), int(y)), (int(x + w), int(y + h)), (b, g, r), 2)
        return
