# -*- coding: utf-8 -*-
"""
LBB : Design : Text Class

@author: kampff
"""

# Imports

# Text Class
class Text:
    def __init__(self, _name, _text, _x, _y, _height, _fill, _size, _family, _anchor):
        self.name = _name           # Name
        self.text = _text           # Text
        self.x = _x                 # X position
        self.y = _y                 # Y position
        self.height = _height       # Height (line)
        self.fill = _fill           # Fill
        self.size = _size           # Size
        self.family = _family       # Family
        self.anchor = _anchor       # Anchor
        return
    
    def draw(self):
        style = f"font-style:normal;font-weight:bold;font-size:{self.size}px;line-height:{self.height};font-family:'{self.family}';white-space:pre;display:inline;fill:#{self.fill};fill-opacity:1;stroke:none"
        tag = f"\t<text class= \"text\" id=\"text_{self.name}\" x=\"{self.x:.3f}\" y=\"{self.y:.3f}\" style=\"{style}\" {self.anchor}>{self.text}</text>\n"
        return tag

#FIN