# -*- coding: utf-8 -*-
"""
Design: SVG Class

@author: kampff
"""

# Import libraries
import numpy as np
import copy

# Import modules
import LBB.Design.profile as Profile
import LBB.Design.text as Text
import LBB.Design.box as Box

# SVG Class
class SVG:
    def __init__(self, _name, _title, _width, _height, _viewbox, _boxes, _with_profile=False, _with_title=False, _with_labels=False):
        self.name = _name                   # Name
        self.title = _title                 # Title
        self.width = _width                 # Width
        self.height = _height               # Height
        self.viewbox = _viewbox             # Viewbox
        self.boxes = _boxes                 # List of boxes
        self.with_profile = _with_profile   # With profile?
        self.with_title = _with_title       # With title?
        self.with_labels = _with_labels     # With labels?
        return
    
    # Create SVG
    def draw(self, output_path):
        svg_file = open(output_path, "w")
        self.write_headers(svg_file)
        self.write_profile(svg_file)
        box_offset = self.get_offset()
        self.write_boxes(box_offset, svg_file)
        self.write_title(svg_file)
        self.write_footer(svg_file)
        svg_file.close()
        return

    # Create animated SVG
    def animate(self, animation_parameters_path, hover, repeat, transform, output_path):
        svg_file = open(output_path, "w")
        self.write_headers(svg_file)
        self.write_profile(svg_file)
        box_offset = self.get_offset()
        self.write_boxes(box_offset, svg_file)
        self.write_title(svg_file)
        #Box,start_time,mid_time,end_time,start_x,mid_x,end_x,start_y,mid_y,end_y,start_delay,value#1,start_value#1,mid_value#1,end_value#1,value#2,start_value#2,mid_value#2,end_value#2,...
        animation_parameters = np.genfromtxt(animation_parameters_path, delimiter=",", dtype=str, comments='##')
        self.write_animation(animation_parameters, hover, repeat, transform, svg_file)
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
    def get_offset(self):
        box_offset = [0.0, 0.0]
        if self.title != None:
            box_offset[0] = 0.0
            box_offset[1] = -7.5
        elif self.with_profile:
            box_offset[0] = 5.0
            box_offset[1] = -11.0
        return box_offset

    # Write boxes
    def write_boxes(self, box_offset, svg_file):
        for box in self.boxes:
            this_box = copy.deepcopy(box)
            this_box.x = this_box.x + box_offset[0]
            this_box.y = this_box.y + box_offset[1]
            tag = this_box.draw()
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
    def write_animation(self, animation_parameters, hover, repeat, transform, svg_file):
        animations = []
        num_boxes = animation_parameters.shape[0]
        svg_file.write("<style>\n")
        for i in range(num_boxes):
            name = animation_parameters[i,0]
            start_time = animation_parameters[i,1]
            mid_time = animation_parameters[i,2]
            end_time = animation_parameters[i,3]
            start_x = animation_parameters[i,4]
            mid_x = animation_parameters[i,5]
            end_x = animation_parameters[i,6]
            start_y = animation_parameters[i,7]
            mid_y = animation_parameters[i,8]
            end_y = animation_parameters[i,9]
            delay = animation_parameters[i,10]
            duration = float(end_time) - float(start_time)
            if hover:
                svg_file.write(f"\t#{self.name}:hover #{name}")
            else:
                svg_file.write(f"\t#{name} ")
            if repeat:
                svg_file.write(f" {{animation: animate_{name} {duration:.2f}s linear infinite; animation-delay: {delay}s}}\n")
            else:
                svg_file.write(f" {{animation: animate_{name} {duration:.2f}s linear; animation-fill-mode: forwards; animation-delay: {delay}s}}\n")
            # Assemble animations
            num_values = (len(animation_parameters[i])-11) // 4
            start_percent = 0
            mid_percent = int(100.0 * (float(mid_time) / float(end_time)))
            end_percent = 100
            animations.append(f"\n")
            animations.append(f"\t@keyframes animate_{name} {{\n")
            animations.append(f"\t\t{start_percent}%     {{")
            if transform:
                start_x = "0.0"
                start_y = "0.0"
                animations.append(f"transform:translate({start_x}px, {start_y}px);")
            for j in range(num_values):
                offset = 11 + (j*4)
                value_name = animation_parameters[i,offset]
                start_value = animation_parameters[i,offset+1]
                animations.append(f"{value_name}:{start_value};")
            animations.append(f"}}\n")
            animations.append(f"\t\t{mid_percent}%    {{")
            if transform:
                animations.append(f"transform:translate({mid_x}px, {mid_y}px);")
            for j in range(num_values):
                offset = 11 + (j*4)
                value_name = animation_parameters[i,offset]
                mid_value = animation_parameters[i,offset+2]
                animations.append(f"{value_name}:{mid_value};")
            animations.append(f"}}\n")
            animations.append(f"\t\t{end_percent}%   {{")
            if transform:
                animations.append(f"transform:translate({end_x}px, {end_y}px);")
            for j in range(num_values):
                offset = 11 + (j*4)
                value_name = animation_parameters[i,offset]
                end_value = animation_parameters[i,offset+3]
                animations.append(f"{value_name}:{end_value};")
            animations.append(f"}}\n")
            animations.append(f"\t}}\n")
        # Write animations
        svg_file.write("".join(animations))
        svg_file.write("</style>\n")
        return

#FIN