# Generate LastBlackBox Logo
import numpy as np
import os

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
boxes_path = repo_path + '/boxes'
logo_path = repo_path + '/course/designs/logo'

# Parameters Class
class Parameters:
    def __init__(self, box_parameters_path, svg_path, box_size, box_style, text, text_size, text_x, text_y, text_fill):
        self.box_parameters_path = box_parameters_path
        self.svg_path = svg_path
        self.box_size = box_size
        self.box_style = box_style
        self.text = text
        self.text_size = text_size
        self.text_x = text_x
        self.text_y = text_y
        self.text_fill = text_fill

# Draw box (add SVG text for a rectangle)
def draw_box(file, name, x, y, width, height, fill, style):
    box_id = 'box_' + name
    line = "\t<rect class=\"box\" id=\"{0}\" transform=\"scale(1,1) translate(1, 1)\" x=\"{1:2f}\" y=\"{2:2f}\" height=\"{3}\" width=\"{4}\" style=\"fill:#{5};{6}\"/>\n".format(box_id, x, y, width, height, fill, style)
    file.write(line)
    return

# Generate logo
def generate_logo(params):
    # Headers
    xml_header = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
    svg_header = "<svg id=\"logo\" width=\"100mm\" height=\"100mm\" viewBox=\"0 0 100 100\" version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:svg=\"http://www.w3.org/2000/svg\">\n"

    # Load box parameters
    box_parameters = np.genfromtxt(params.box_parameters_path, delimiter=",", dtype=str)
    num_boxes = box_parameters.shape[0]

    # Params
    if params.text != '':
        box_offset = -7.5
    else:
        box_offset = 0.0

    # Text
    text_style = "font-style:normal;font-weight:bold;font-size:{0};line-height:1.25;font-family:'Liberation Mono';white-space:pre;display:inline;fill:#{1};fill-opacity:1;stroke:none".format(params.text_size, params.text_fill)
    text_tag = "\t<text class= \"text\" id=\"nbb\" x=\"{0}\" y=\"{1}\" style=\"{2}\">{3}</text>\n".format(params.text_x, params.text_y, text_style, params.text)

    # Open SVG ouput
    svg_file = open(params.svg_path, "w")

    # Write headers
    ret = svg_file.write(xml_header)
    ret = svg_file.write(svg_header)

    for i in range(num_boxes):
        name = box_parameters[i,0]
        x = float(box_parameters[i,1])
        y = float(box_parameters[i,2])+box_offset
        fill = box_parameters[i,3]
        draw_box(svg_file, name, x, y, params.box_size, params.box_size, fill, params.box_style)

    # Add text?
    if params.text != '':
        ret = svg_file.write(text_tag)

    # Close SVG output
    ret = svg_file.write("</svg>")
    svg_file.close()
    return

# Generate Logos
# --------------

# Create Parameters List
parameters = []

# LBB Parameters
box_parameters_path = logo_path + "/box_parameters_LBB.csv"
svg_path = logo_path + "/svg/logo_LBB.svg"
box_size = 13.0
box_stroke = 1.0
box_style = "fill-opacity:1;stroke:#FFFFFF;stroke-width:{0};stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1".format(box_stroke)
text = ''
text_size = 8.5
text_x = 4.5
text_y = 86.5
text_fill = 'FFFFFF'
parameters.append(Parameters(box_parameters_path, svg_path, box_size, box_style, text, text_size, text_x, text_y, text_fill))
# ...with white text
svg_path = logo_path + "/svg/logo_LBB_white_text.svg"
text_fill = 'FFFFFF'
text = "The Last Black Box"
parameters.append(Parameters(box_parameters_path, svg_path, box_size, box_style, text, text_size, text_x, text_y, text_fill))
# ...with black text
svg_path = logo_path + "/svg/logo_LBB_black_text.svg"
text_fill = '000000'
parameters.append(Parameters(box_parameters_path, svg_path, box_size, box_style, text, text_size, text_x, text_y, text_fill))

# NBB Parameters
box_parameters_path = logo_path + "/box_parameters_NBB.csv"
svg_path = logo_path + "/svg/logo_NBB.svg"
text = ''
text_size = 9.5
box_size = 13.0
box_stroke = 1.0
box_style = "fill-opacity:1;stroke:#000000;stroke-width:{0};stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1".format(box_stroke)
text_x = 10.5
text_y = 86.5
text_fill = 'FFFFFF'
parameters.append(Parameters(box_parameters_path, svg_path, box_size, box_style, text, text_size, text_x, text_y, text_fill))
# ...with white text
svg_path = logo_path + "/svg/logo_NBB_white_text.svg"
text = "No Black Boxes"
text_fill = 'FFFFFF'
parameters.append(Parameters(box_parameters_path, svg_path, box_size, box_style, text, text_size, text_x, text_y, text_fill))
# ...with black text
svg_path = logo_path + "/svg/logo_NBB_black_text.svg"
text_fill = '000000'
parameters.append(Parameters(box_parameters_path, svg_path, box_size, box_style, text, text_size, text_x, text_y, text_fill))

# Generate (and convert)
for p in parameters:
    print(p.svg_path)
    generate_logo(p)
    
    # Convert to PNG
    png_path = logo_path + "/png/" + p.svg_path.split('/')[-1][:-3] + "png"
    os.system(f"inkscape -w 1024 -h 1024 {p.svg_path} -o {png_path}")
    

#FIN