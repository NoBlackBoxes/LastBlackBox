# -*- coding: utf-8 -*-
"""
LBB : Design : Arrow Class

@author: kampff
"""

# Imports

# Arrow Class
class Arrow:
    def __init__(self, _parent, _fill, _state):
        self.parent = _parent   # Parent (box)
        self.fill = _fill       # Fill
        self.state = _state     # State (direction)
        return
    
    def draw(self):
        id = 'arrow_' + self.parent.name
        style = f"fill:#{self.fill};stroke:none"
        half_width = (self.parent.width + self.parent.stroke) / 2.0
        half_height = (self.parent.height + self.parent.stroke) / 2.0
        if self.state == 1:
            points = f"{self.parent.x+self.parent.width+self.parent.stroke/2.0}, {self.parent.y+half_height+0.5}, {self.parent.x+self.parent.width+self.parent.stroke/2.0}, {self.parent.y+half_height-0.5}, {self.parent.x+self.parent.width+1.2}, {self.parent.y+half_height}"
        elif self.state == -1:
            points = f"{self.parent.x-self.parent.stroke/2.0}, {self.parent.y+half_height+0.5}, {self.parent.x-self.parent.stroke/2.0}, {self.parent.y+half_height-0.5}, {self.parent.x-1.2}, {self.parent.y+half_height}"
        else:
            points = f"{self.parent.x+half_width-0.5}, {self.parent.y+self.parent.height+self.parent.stroke/2.0}, {self.parent.x+half_width+0.5}, {self.parent.y+self.parent.height+self.parent.stroke/2.0}, {self.parent.x+half_width}, {self.parent.y+self.parent.height+1.2}"
        tag = f"\t<polygon class=\"arrow\" id=\"{id}\" style=\"{style}\" points=\"{points}\"/>\n"
        return tag

#FIN