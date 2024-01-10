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
svg_path = layout_path + "/layout.svg"

# Load box parameters
box_parameters = np.genfromtxt(box_parameters_path, delimiter=",", dtype=str)
num_boxes = box_parameters.shape[0]

# Defaults
box_size = 13.0
box_stroke = 0.125
num_rows = 4
num_cols = 7

# Draw box (add SVG text for a rectangle with text name)
def draw_box(file, name, x, y, width, height, fill, arrow_state):
    text_fill = "FFFFFF"
    text_id = 'text_' + name
    text_fontsize = 1.75
    text_style = "font-style:normal;font-weight:bold;font-size:{0};line-height:1.00;font-family:'Arial';white-space:pre;display:inline;fill:#{1};fill-opacity:1;stroke:none".format(text_fontsize, text_fill)
    text_tag = "\t<text class= \"text\" id=\"{0}\" x=\"{1}\" y=\"{2}\" style=\"{3}\" alignment-baseline=\"middle\" text-anchor=\"middle\">{4}</text>\n".format(text_id, x+width/2.0, y+text_fontsize/3.0+height/2.0, text_style, name)
    box_fill = fill
    box_id = 'box_' + name
    box_style = "fill:#{0};fill-opacity:1;stroke:#FFFFFF;stroke-width:{1};stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1".format(box_fill, box_stroke)
    box_tag = "\t<rect class=\"box\" id=\"{0}\" transform=\"scale(1,1) translate(0, 0)\" x=\"{1:2f}\" y=\"{2:2f}\" width=\"{3:2f}\" height=\"{4:2f}\" style=\"{5}\"/>\n".format(box_id, x, y, width, height, box_style)
    arrow_fill = "DDDDDD"
    arrow_id = 'arrow_' + name
    arrow_style = "fill:#{0};stroke:none".format(arrow_fill)
    half_width = (width + box_stroke) / 2.0
    half_height = (height + box_stroke) / 2.0
    # Draw arrow (in particular direction, 1 right, -1 left, 0 down)
    if arrow_state == 1:
        arrow_points = "{0},{1} {2},{3} {4},{5}".format(x+width+box_stroke/2.0, y+half_height+0.5, x+width+box_stroke/2.0, y+half_height-0.5, x+width+1.2, y+half_height)
    elif arrow_state == -1:
        arrow_points = "{0},{1} {2},{3} {4},{5}".format(x-box_stroke/2.0, y+half_height+0.5, x-box_stroke/2.0, y+half_height-0.5, x-1.2, y+half_height)
    else:
        arrow_points = "{0},{1} {2},{3} {4},{5}".format(x+half_width-0.5, y+height+box_stroke/2.0, x+half_width+0.5, y+height+box_stroke/2.0, x+half_width, y+height+1.2)
    arrow_tag = "\t<polygon class=\"arrow\" id=\"{0}\" style=\"{1}\" points=\"{2}\"/>\n".format(arrow_id, arrow_style, arrow_points) 
    file.write("<g id=\"{0}\">\n".format(name))
    file.write(box_tag)
    file.write(text_tag)
    if(arrow_state != 2):
        file.write(arrow_tag)
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
    arrow_state = float(box_parameters[i,4])
    draw_box(svg_file, name, x, y, box_size, box_size, fill, arrow_state)

# Close SVG output
ret = svg_file.write("</svg>")
svg_file.close()

#FIN