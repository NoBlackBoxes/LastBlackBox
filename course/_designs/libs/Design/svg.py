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
    
    # Create SVG
    def draw(self, box_parameters_path, output_path):
        svg_file = open(output_path, "w")
        self.write_headers(svg_file)
        self.write_profile(svg_file)
        box_offset = self.set_offset()
        box_parameters = np.genfromtxt(box_parameters_path, delimiter=",", dtype=str)
        self.write_boxes(box_parameters, box_offset, svg_file)
        self.write_title(svg_file)
        self.write_footer(svg_file)
        svg_file.close()
        return

    # Create animated SVG
    def animate(self, box_parameters_path, animation_parameters_path, output_path):
        svg_file = open(output_path, "w")
        self.write_headers(svg_file)
        self.write_profile(svg_file)
        box_offset = self.set_offset()
        box_parameters = np.genfromtxt(box_parameters_path, delimiter=",", dtype=str)
        self.write_boxes(box_parameters, box_offset, svg_file)
        self.write_title(svg_file)
        #Box,start_time,mid_time,end_time,start_delay,value#1,start_value#1,mid_value#1,end_value#1,value#2,start_value#2,mid_value#2,end_value#2,...
        animation_parameters = np.genfromtxt(animation_parameters_path, delimiter=",", dtype=str)
        self.write_animation(animation_parameters, svg_file)
        self.write_footer(svg_file)
        svg_file.close()
        return

    # Write XML and SVG headers
    def write_headers(self, svg_file):
        xml_header = f"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
        svg_header = f"<svg id=\"{self.name}\" width=\"{self.width}mm\" height=\"{self.height}mm\" viewBox=\"{self.viewbox}\" version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:svg=\"http://www.w3.org/2000/svg\">\n"
        ret = svg_file.write(xml_header)
        ret = svg_file.write(svg_header)
        return

    # Write profile (if included)
    def write_profile(self, svg_file):
        if self.with_profile:
            tag = Profile.Profile().draw()
            ret = svg_file.write(tag)
        return

    # Set box offset
    def set_offset(self):
        box_offset = [0.0, 0.0]
        if self.title != None:
            box_offset[0] = 0.0
            box_offset[1] = -7.5
        elif self.with_profile:
            box_offset[0] = 5.0
            box_offset[1] = -11.0
        return box_offset

    # Write boxes (based on parameters)
    def write_boxes(self, box_parameters, box_offset, svg_file):
        num_boxes = box_parameters.shape[0]
        for i in range(num_boxes):
            name = box_parameters[i,0]
            x = float(box_parameters[i,1])+box_offset[0]
            y = float(box_parameters[i,2])+box_offset[1]
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
        return

    # Write title (if included)
    def write_title(self, svg_file):
        if self.title != None:
            tag = self.title.draw()
            ret = svg_file.write(tag)
        return

    # Write SVG footer
    def write_footer(self, svg_file):
        ret = svg_file.write("</svg>")
        return

    # Write animation
    def write_animation(self, animation_parameters, svg_file):
        svg_file.write("<style>\n")
        num_boxes = animation_parameters.shape[0]
        for i in range(num_boxes):
            name = animation_parameters[i,0]
            svg_file.write(f"\t#box_{name}")
            svg_file.write("\t")
            if len(name) < 7:
                svg_file.write("\t")
            if len(name) < 11:
                svg_file.write("\t")
            svg_file.write(f" {{animation: animate_{name} 0.6s linear; animation-fill-mode: forwards; animation-delay: 1s}}\n")
        for i in range(num_boxes):
            name = animation_parameters[i,0]
            svg_file.write(f"\n")
            svg_file.write(f"\t@keyframes animate_{name} {{\n")
            svg_file.write(f"\t\t0%   {{fill:black;\tstroke:white}}\n")
            svg_file.write(f"\t\t95%  {{fill:white;\tstroke:black}}\n")
            svg_file.write(f"\t\t100% {{fill:white;\tstroke:black}}\n")
            svg_file.write(f"\t}}\n")
        svg_file.write("</style>\n")
        return

#FIN