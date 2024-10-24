# -*- coding: utf-8 -*-
"""
LBB: Image Class

@author: kampff
"""

# Import libraries

# Import modules

# Image Class
class Image:
    def __init__(self, text=None):
        self.html = None        # html
        if text:
            self.parse(text)
        return
    
    def parse(self, text):
        self.html = "<img src=\"https://raw.githubusercontent.com/NoBlackBoxes/LastBlackBox/master" + text[36:-4] + '\n'
        return

    def render(self):
        return "<p align=\"center\">\n" + self.html + "</p>\n"

#FIN