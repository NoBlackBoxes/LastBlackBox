# -*- coding: utf-8 -*-
"""
Design: PNG Class

@author: kampff
"""

# Import libraries
import os

# Import modules

# PNG Class
class PNG:
    def __init__(self, _name, width=None, height=None, dpi=None):
        self.name = _name           # Name
        self.width = width         # Width
        self.height = height       # Height
        self.dpi = dpi             # DPI
        return
    
    def convert(self, svg_path, output_path):
        if self.dpi == None:
            os.system(f"inkscape -w {self.width} -h {self.height} {svg_path} -o {output_path}")
        else:
            os.system(f"inkscape -D --export-dpi={self.dpi} {svg_path} -o {output_path}")
        return

#FIN