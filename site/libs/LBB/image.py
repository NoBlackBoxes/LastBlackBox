# -*- coding: utf-8 -*-
"""
LBB: Image Class

@author: kampff
"""

# Import libraries
import os

# Import modules

# Image Class
class Image:
    def __init__(self, text=None):
        self.html = None        # html
        if text:
            self.parse(text)
        return
    
    def parse(self, text):
        self.html = "<img src=\"../../../boxes" + text[12:]
        return

    def render(self):
        return "<p align=\"center\">\n" + self.html + "\n</p>"

#FIN