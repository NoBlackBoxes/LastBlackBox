# -*- coding: utf-8 -*-
"""
Design: SVG Class

@author: kampff
"""

# Import libraries
import numpy as np

# Import modules
import Design.profile as Profile
import Design.text as Text
import Design.box as Box

# SVG Class
class SVG:
    def __init__(self, _name, _title, _width, _height, _viewbox, with_profile=False, with_title=False, with_labels=False):
        self.name = _name                   # Name
        self.title = _title                 # Title
        self.width = _width                 # Width
        self.height = _height               # Height
        self.viewbox = _viewbox             # Viewbox
        self.boxes = []                     # List of boxes
        self.with_profile = with_profile    # With profile?
        self.with_labels = with_labels      # With labels?
        return
    
    def draw(self, box_parameters_path, output_path):
        svg_file = open(output_path, "w")

        # Write headers
        xml_header = f"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
        svg_header = f"<svg id=\"{self.name}\" width=\"{self.width}mm\" height=\"{self.height}mm\" viewBox=\"{self.viewbox}\" version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:svg=\"http://www.w3.org/2000/svg\">\n"
        ret = svg_file.write(xml_header)
        ret = svg_file.write(svg_header)

        # Add profile?
        if self.with_profile:
            tag = Profile.Profile().draw()
            ret = svg_file.write(tag)

        # Load boxes
        box_parameters = np.genfromtxt(box_parameters_path, delimiter=",", dtype=str)
        num_boxes = box_parameters.shape[0]

        # Set offset
        box_offset_x = 0.0
        box_offset_y = 0.0
        if self.title != None:
            box_offset_x = 0.0
            box_offset_y = -7.5
        elif self.with_profile:
            box_offset_x = 5.0
            box_offset_y = -11.0

        # Draw boxes
        for i in range(num_boxes):
            name = box_parameters[i,0]
            x = float(box_parameters[i,1])+box_offset_x
            y = float(box_parameters[i,2])+box_offset_y
            width = float(box_parameters[i,3])
            height = float(box_parameters[i,4])
            stroke = float(box_parameters[i,5])
            fill = box_parameters[i,6]
            border = box_parameters[i,7]
            state = float(box_parameters[i,8])
            if self.with_labels:
                label = name
            else:
                label = None
            box = Box.Box(name, label, state, x, y, width, height, stroke, fill, border)
            tag = box.draw()
            ret = svg_file.write(tag)

        # Add title?
        if self.title != None:
            tag = self.title.draw()
            ret = svg_file.write(tag)
        
        # Close SVG output
        ret = svg_file.write("</svg>")
        svg_file.close()

        return

#FIN