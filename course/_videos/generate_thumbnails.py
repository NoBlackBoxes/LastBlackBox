# Generate Video Thumbnails

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

# Specify paths
boxes_path = "/home/kampff/NoBlackBoxes/LastBlackBox/course/_videos/boxes"

# - Find all boxes at "boxes_path"
# - Find all *.md files in each box
# -- Is an NB3 video?
# - Open template SVG file and swap Box Title/Subtitle
# - Save PNG thumbnail into Box 

#FIN