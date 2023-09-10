# Generate LastBlackBox Layout
import numpy as np
import os

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
boxes_path = repo_path + '/boxes'
layout_path = repo_path + '/course/designs/layout'
svg_path = layout_path + "/output.svg"

# List all "boxes" in order of opening
boxes = [
    ('atoms', 'Atoms'),
    ('electrons', 'Electrons'),
    ('magnets', 'Magnets'),
    ('light', 'Light'),
    ('sensors', 'Sensors'),
    ('motors', 'Motors'),
    ('transistors','Transistors'),
    ('amplifiers', 'Amplifiers'),
    ('reflexes', 'Reflexes'),
    ('power', 'Power'),
    ('data', 'Data'),
    ('logic', 'Logic'),
    ('memory', 'Memory'),
    ('fpgas', 'FPGAs'),
    ('computers', 'Computers'),
    ('control', 'Control'),
    ('behaviour', 'Behaviour'),
    ('systems', 'Systems'),
    ('linux', 'Linux'),
    ('python', 'Python'),
    ('networks', 'Networks'),
    ('websites', 'Websites'),
    ('servers', 'Servers'),
    ('security', 'Security'),
    ('audio', 'Audio'),
    ('vision', 'Vision'),
    ('learning', 'Learning'),
    ('intelligence', 'Intelligence')
]
num_boxes = len(boxes)

# Defaults
box_size = 11
box_stroke = 0.25
num_rows = 4
num_cols = 7

# Generate box parameters
box_parameters = []
for i in range(num_boxes):
    x = box_size * (i%num_cols)
    y = box_size * (i // num_cols)
    x = x + ((i%num_cols) * 0.75)
    y = y + ((i // num_cols) * 0.75)
    x = x + 1.00
    y = y + 1.00
    box_parameters.append((x,y))

# Draw box (add SVG text for a rectangle with text name)
def draw_box(file, name, x, y, width, height):
    text_fill = "ffffff"
    text_id = 'text_' + name
    text_fontsize = 1.25
    text_style = "font-style:normal;font-weight:bold;font-size:{0};line-height:1.00;font-family:'Liberation Mono';white-space:pre;display:inline;fill:#{1};fill-opacity:1;stroke:none".format(text_fontsize, text_fill)
    text_tag = "\t<text class= \"text\" id=\"{0}\" x=\"{1}\" y=\"{2}\" style=\"{3}\" alignment-baseline=\"middle\" text-anchor=\"middle\">{4}</text>\n".format(text_id, x+width/2.0, y+height/2.0, text_style, name)
    box_fill = '000000'
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
svg_header = "<svg id=\"layout\" width=\"83.5mm\" height=\"48.25mm\" viewBox=\"0 0 83.5 48.125\" version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:svg=\"http://www.w3.org/2000/svg\">\n"

# Open SVG ouput
svg_file = open(svg_path, "w")

# Write headers
ret = svg_file.write(xml_header)
ret = svg_file.write(svg_header)

# Draw boxes
for i in range(num_boxes):
    name = boxes[i][1]
    id = "box_{0}".format(name)
    x = float(box_parameters[i][0])
    y = float(box_parameters[i][1])
    draw_box(svg_file, name, x, y, box_size, box_size)

# Close SVG output
ret = svg_file.write("</svg>")
svg_file.close()

#FIN