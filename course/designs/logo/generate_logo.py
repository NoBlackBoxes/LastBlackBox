# Generate LastBlackBox Logo
import numpy as np
import os

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
boxes_path = repo_path + '/boxes'
logo_path = repo_path + '/course/designs/logo'

## LBB Defaults
#box_parameters_path = logo_path + "/box_parameters_LBB.csv"
#svg_path = logo_path + "/logo_LBB.svg"
#with_text = False
#box_size = 13.0
#box_stroke = 1.0
#box_style = "fill-opacity:1;stroke:#FFFFFF;stroke-width:{0};stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1".format(box_stroke)

# NBB Defaults
box_parameters_path = logo_path + "/box_parameters_NBB.csv"
svg_path = logo_path + "/logo_NBB.svg"
with_text = False
box_size = 13.0
box_stroke = 1.0
box_style = "fill-opacity:1;stroke:#000000;stroke-width:{0};stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1".format(box_stroke)

# Load box parameters
box_parameters = np.genfromtxt(box_parameters_path, delimiter=",", dtype=str)
num_boxes = box_parameters.shape[0]

# Params
if with_text:
    offset = -7.5
else:
    offset = 0.0

# Draw box (add SVG text for a rectangle)
def draw_box(file, name, x, y, width, height, fill, style):
    box_id = 'box_' + name
    line = "\t<rect class=\"box\" id=\"{0}\" transform=\"scale(1,1) translate(1, 1)\" x=\"{1:2f}\" y=\"{2:2f}\" height=\"{3}\" width=\"{4}\" style=\"fill:#{5};{6}\"/>\n".format(box_id, x, y, width, height, fill, style)
    file.write(line)
    return

# Headers
xml_header = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
svg_header = "<svg id=\"logo\" width=\"100mm\" height=\"100mm\" viewBox=\"0 0 100 100\" version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:svg=\"http://www.w3.org/2000/svg\">\n"

# Text
text_x = 10.5
text_y = 94.0 + offset
#text_fill = "000000"
text_fill = "ffffff"
text_style = "font-style:normal;font-weight:bold;font-size:9.5;line-height:1.25;font-family:'Liberation Mono';white-space:pre;display:inline;fill:#{0};fill-opacity:1;stroke:none".format(text_fill)
text_tag = "\t<text class= \"text\" id=\"nbb\" x=\"{0}\" y=\"{1}\" style=\"{2}\">The Last Black Box</text>\n".format(text_x, text_y, text_style)

# Open SVG ouput
svg_file = open(svg_path, "w")

# Write headers
ret = svg_file.write(xml_header)
ret = svg_file.write(svg_header)

for i in range(num_boxes):
    name = box_parameters[i,0]
    x = float(box_parameters[i,1])
    y = float(box_parameters[i,2])
    fill = box_parameters[i,3]
    draw_box(svg_file, name, x, y, box_size, box_size, fill, box_style)

# Add text?
if(with_text):
    ret = svg_file.write(text_tag)

# Close SVG output
ret = svg_file.write("</svg>")
svg_file.close()

#FIN