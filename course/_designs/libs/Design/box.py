# -*- coding: utf-8 -*-
"""
Design: Box Class

@author: kampff
"""

# Import libraries

# Import modules
import Design.text as Text
import Design.arrow as Arrow

# Box Class
class Box:
    def __init__(self, _name, _label, _state, _x, _y, _width, _height, _stroke, _fill, _border):
        self.name = _name               # Name
        self.label = _label             # Label
        self.state = _state             # State (arrow direction)
        self.x = _x                     # X position
        self.y = _y                     # Y position
        self.width = _width             # Width
        self.height = _height           # Height
        self.stroke = _stroke           # Stroke
        self.fill = _fill               # Fill
        self.border = _border           # Border
        return
    
    def draw(self):
        tag = []
        tag.append(f"<g id=\"{self.name}\">\n")
        style = f"fill:#{self.fill};fill-opacity:1;stroke:#{self.border};stroke-width:{self.stroke};stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1"
        rect = f"\t<rect class=\"box\" id=\"box_{self.name}\" transform=\"scale(1,1) translate(0, 0)\" x=\"{self.x:2f}\" y=\"{self.y:2f}\" width=\"{self.width}\" height=\"{self.height}\" style=\"{style}\"/>\n"
        tag.append(rect)
        if self.label != None:
            anchor =  "alignment-baseline=\"middle\" text-anchor=\"middle\""
            label = Text.Text(self.name, self.label, self.x+self.width/2.0, self.y+1.75/3.0+self.height/2.0, 1.00, "FFFFFF", 1.75, "Arial", anchor)
            tag.append(label.draw())
        if self.state != 0:
            arrow = Arrow.Arrow(self, "DDDDDD", self.state)
            tag.append(arrow.draw())
        tag.append(f"</g>\n")
        return "".join(tag)

#FIN