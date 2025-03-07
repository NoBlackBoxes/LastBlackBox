# -*- coding: utf-8 -*-
"""
Design: Box Class

@author: kampff
"""

# Import libraries

# Import modules
import LBB.Design.label as Label
import LBB.Design.arrow as Arrow

# Box Class
class Box:
    def __init__(self, _name, _label, _label_size, _arrow, _x, _y, _width, _height, _stroke, _fill, _border):
        self.name = _name               # Name
        self.label = _label             # Label
        self.label_size = _label_size   # Label size
        self.arrow = _arrow             # Arrow (0 = none, 1 = right, -1 = left)
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
        tag.append(f"<g id=\"{self.name}\" transform=\"scale(1,1) translate(0, 0)\" style=\"fill:{self.fill};stroke:{self.border};font-size:{self.label_size}px;\">\n")
        style = f"fill-opacity:1;stroke-width:{self.stroke};stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1"
        rect = f"\t<rect class=\"box\" id=\"box_{self.name}\" x=\"{self.x:.3f}\" y=\"{self.y:.3f}\" width=\"{self.width:.1f}\" height=\"{self.height:.1f}\" style=\"{style}\"/>\n"
        tag.append(rect)
        if self.label != None:
            anchor =  "alignment-baseline=\"middle\" text-anchor=\"middle\""
            if (self.fill == "#000000"):
                label = Label.Label(self.name, self.label, self.x+self.width/2.0, self.y+1.75/3.0+self.height/2.0, 1.00, "#FFFFFF", "Arial", anchor)
            else:
                label = Label.Label(self.name, self.label, self.x+self.width/2.0, self.y+1.75/3.0+self.height/2.0, 1.00, "#000000", "Arial", anchor)
            tag.append(label.draw())
        if self.arrow != 0:
            arrow = Arrow.Arrow(self, "DDDDDD", self.arrow)
            tag.append(arrow.draw())
        tag.append(f"</g>\n")
        return "".join(tag)

#FIN