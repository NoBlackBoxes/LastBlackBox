# Generate LastBlackBox Layout
import numpy as np
import os

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
boxes_path = repo_path + '/boxes'
layout_path = repo_path + '/course/designs/layout'
box_parameters_path = layout_path + "/box_parameters.csv"
svg_path = layout_path + "/output.svg"

# Load box parameters
box_parameters = np.genfromtxt(box_parameters_path, delimiter=",", dtype=str)
num_boxes = box_parameters.shape[0]

# Defaults
box_size = 13.0
box_stroke = 0.125
num_rows = 4
num_cols = 7

# Draw box (add SVG text for a rectangle with text name)
def draw_box(file, name, x, y, width, height, fill):
    text_fill = "ffffff"
    text_id = 'text_' + name
    text_fontsize = 1.75
    text_style = "font-style:normal;font-weight:bold;font-size:{0};line-height:1.00;font-family:'Arial';white-space:pre;display:inline;fill:#{1};fill-opacity:1;stroke:none".format(text_fontsize, text_fill)
    text_tag = "\t<text class= \"text\" id=\"{0}\" x=\"{1}\" y=\"{2}\" style=\"{3}\" alignment-baseline=\"middle\" text-anchor=\"middle\">{4}</text>\n".format(text_id, x+width/2.0, y+height/2.0, text_style, name)
    box_fill = fill
    box_id = 'box_' + name
    box_style = "fill-opacity:1;stroke:#FFFFFF;stroke-width:{0};stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1".format(box_stroke)
    box_tag = "\t<rect class=\"box\" id=\"{0}\" transform=\"scale(1,1) translate(0, 0)\" x=\"{1:2f}\" y=\"{2:2f}\" width=\"{3}\" height=\"{4}\" style=\"fill:#{5};{6}\"/>\n".format(box_id, x, y, width, height, box_fill, box_style)
    file.write("<g id=\"{0}\">\n".format(name))
    file.write(box_tag)
    file.write(text_tag)
    file.write("</g>\n")
    return

# Headers
xml_header = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
#svg_header = "<svg id=\"layout\" width=\"100mm\" height=\"100mm\" viewBox=\"0 0 100 100\" version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:svg=\"http://www.w3.org/2000/svg\">\n"
svg_header = "<svg id=\"layout\" width=\"100mm\" height=\"60mm\" viewBox=\"0 0 100 60\" version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:svg=\"http://www.w3.org/2000/svg\">\n"

# Open SVG ouput
svg_file = open(svg_path, "w")

# Write headers
ret = svg_file.write(xml_header)
ret = svg_file.write(svg_header)

# Draw boxes
for i in range(num_boxes):
    name = box_parameters[i,0]
    x = float(box_parameters[i,1])
    y = float(box_parameters[i,2])
    fill = box_parameters[i,3]
    draw_box(svg_file, name, x, y, box_size, box_size, fill)

# Close SVG output
ret = svg_file.write("</svg>")
svg_file.close()

#FIN