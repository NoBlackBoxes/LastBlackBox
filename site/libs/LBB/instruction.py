# -*- coding: utf-8 -*-
"""
LBB: Instruction Class

@author: kampff
"""

# Import libraries
import os

# Import modules

# Instruction Class
class Instruction:
    def __init__(self, text=None):
        self.html = None     # instruction text
        if text:
            self.parse(text)
        return
    
    def parse(self, text):
        self.html = text
        # TO DO: convert emphasis tags
        return

    def render(self):
        return f"<h4 id=\"instruction\">{self.html}</h4>"
#FIN