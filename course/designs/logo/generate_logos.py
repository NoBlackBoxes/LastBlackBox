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
    def __init__(self, width, height, viewbox, box_parameters_path, svg_path, box_size, box_style, text, text_size, text_x, text_y, text_fill, profile):
        self.width = width
        self.height = height
        self.viewbox = viewbox
        self.box_parameters_path = box_parameters_path
        self.svg_path = svg_path
        self.box_size = box_size
        self.box_style = box_style
        self.text = text
        self.text_size = text_size
        self.text_x = text_x
        self.text_y = text_y
        self.text_fill = text_fill
        self.profile = profile

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
    svg_header = f"<svg id=\"logo\" width=\"{params.width}\" height=\"{params.height}\" viewBox=\"{params.viewbox}\" version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:svg=\"http://www.w3.org/2000/svg\">\n"

    # Load box parameters
    box_parameters = np.genfromtxt(params.box_parameters_path, delimiter=",", dtype=str)
    num_boxes = box_parameters.shape[0]

    # Params
    box_offset_x = 0.0
    box_offset_y = 0.0
    if params.text != '':
        box_offset_x = 0.0
        box_offset_y = -7.5
    if params.profile:
        box_offset_x = 5.0
        box_offset_y = -11.0

    # Text
    text_style = "font-style:normal;font-weight:bold;font-size:{0};line-height:1.25;font-family:'Liberation Mono';white-space:pre;display:inline;fill:#{1};fill-opacity:1;stroke:none".format(params.text_size, params.text_fill)
    text_tag = "\t<text class= \"text\" id=\"nbb\" x=\"{0}\" y=\"{1}\" style=\"{2}\">{3}</text>\n".format(params.text_x, params.text_y, text_style, params.text)

    # Profile
    profile_style = "fill-opacity:0;stroke:#FFFFFF;stroke-width:0.25"
    profile_tag = "<path class=\"profile\" id=\"head\" d=\"M 36.766274,124.56293 C 36.43594,122.21923 35.11219,102.19327 33.858365,100.59719 30.967333,98.039539 26.56689,98.31006 23.080291,99.23721 20.097634,99.082276 16.885914,99.642988 14.238413,97.877226 10.985701,94.274373 11.371495,89.316454 12.058688,84.783991 L 11.453474,83.424006 9.7584006,82.92969 C 8.9096572,81.766448 8.8734894,79.949036 9.2737493,78.606266 L 9.8789605,78.111949 8.6685371,77.617633 7.5786737,76.506036 C 7.95241,74.270546 9.1797114,71.811261 9.6378407,69.588067 6.9951628,68.481388 2.3994114,68.695347 1.1600537,65.75896 0.51626267,63.425099 12.574685,47.188897 13.150962,45.253439 9.6981207,21.388535 22.583584,4.1120555 46.213365,1.7757344 63.856132,0.65184101 84.621403,4.6358832 94.538679,21.169658 c 6.604281,12.325939 5.299826,27.300526 -0.605212,39.653516 -3.214133,6.049842 -8.007602,12.175921 -10.416393,18.65368 -3.460075,14.708985 3.373272,29.813916 6.662151,43.976936\" style=\"{0}\"/>\n".format(profile_style)

    # Open SVG ouput
    svg_file = open(params.svg_path, "w")

    # Write headers
    ret = svg_file.write(xml_header)
    ret = svg_file.write(svg_header)

    # Add profile?
    if params.profile:
        ret = svg_file.write(profile_tag)

    for i in range(num_boxes):
        name = box_parameters[i,0]
        x = float(box_parameters[i,1])+box_offset_x
        y = float(box_parameters[i,2])+box_offset_y
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
width = "100mm"
height = "100mm"
viewbox = "0 0 100 100"
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
profile = False
parameters.append(Parameters(width, height, viewbox, box_parameters_path, svg_path, box_size, box_style, text, text_size, text_x, text_y, text_fill, profile))
# ...with white text
svg_path = logo_path + "/svg/logo_LBB_white_text.svg"
text_fill = 'FFFFFF'
text = "The Last Black Box"
parameters.append(Parameters(width, height, viewbox, box_parameters_path, svg_path, box_size, box_style, text, text_size, text_x, text_y, text_fill, profile))
# ...with black text
svg_path = logo_path + "/svg/logo_LBB_black_text.svg"
text_fill = '000000'
parameters.append(Parameters(width, height, viewbox, box_parameters_path, svg_path, box_size, box_style, text, text_size, text_x, text_y, text_fill, profile))

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
profile = False
parameters.append(Parameters(width, height, viewbox, box_parameters_path, svg_path, box_size, box_style, text, text_size, text_x, text_y, text_fill, profile))
# ...with white text
svg_path = logo_path + "/svg/logo_NBB_white_text.svg"
text = "No Black Boxes"
text_fill = 'FFFFFF'
parameters.append(Parameters(width, height, viewbox, box_parameters_path, svg_path, box_size, box_style, text, text_size, text_x, text_y, text_fill, profile))
# ...with black text
svg_path = logo_path + "/svg/logo_NBB_black_text.svg"
text_fill = '000000'
parameters.append(Parameters(width, height, viewbox, box_parameters_path, svg_path, box_size, box_style, text, text_size, text_x, text_y, text_fill, profile))

# Profile Parameters
width = "100 mm"
height = "125 mm"
viewbox = "0 0 100 125"
box_parameters_path = logo_path + "/box_parameters_NBB.csv"
svg_path = logo_path + "/svg/logo_NBB_profile.svg"
text = ''
text_size = 9.5
box_size = 13.0
box_stroke = 0.5
box_style = "fill-opacity:1;stroke:#000000;stroke-width:{0};stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1".format(box_stroke)
text_x = 10.5
text_y = 86.5
text_fill = 'FFFFFF'
profile = True
parameters.append(Parameters(width, height, viewbox, box_parameters_path, svg_path, box_size, box_style, text, text_size, text_x, text_y, text_fill, profile))

# Generate (and convert)
for p in parameters:
    print(p.svg_path)
    generate_logo(p)
    
    # Convert to PNG
    png_path = logo_path + "/png/" + p.svg_path.split('/')[-1][:-3] + "png"
    os.system(f"inkscape -w {int(p.width[:-2])*10} -h {int(p.height[:-2])*10} {p.svg_path} -o {png_path}")
    
#FIN