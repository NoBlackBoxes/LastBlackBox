# Generate LBB layouts

#----------------------------------------------------------
# Set library paths
import sys
libs_path = "/home/kampff/NoBlackBoxes/LastBlackBox/course/_designs/libs"
sys.path.append(libs_path)
#----------------------------------------------------------

# Import Libraries
import Design.utilities as Utilities

# Import Modules
import Design.svg as SVG
import Design.png as PNG
import Design.text as Text

# Specify output folders
base_path = "/home/kampff/NoBlackBoxes/LastBlackBox/course/_designs"
output_folder = f"{base_path}/layout"
svg_folder = f"{output_folder}/svg"
png_folder = f"{output_folder}/png"

# Clear/create folders
Utilities.clear_folder(svg_folder)
Utilities.clear_folder(png_folder)

# ---
# LBB
# ---
box_parameters_path = f"{output_folder}/box_parameters.csv"
svg = SVG.SVG("layout_LBB", None, 98.75, 56.0, "0 0 98.75 56.0", with_profile=False, with_title=False, with_labels=True)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(box_parameters_path, svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, dpi=600).convert(svg_path, png_path)

#FIN