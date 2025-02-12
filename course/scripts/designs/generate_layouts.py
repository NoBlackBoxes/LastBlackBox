# Generate Course Layouts

# Import Libraries
import LBB.Design.utilities as Utilities

# Import Modules
import LBB.Engine.config as Config
import LBB.Design.layout as Layout
import LBB.Design.logo as Logo
import LBB.Design.svg as SVG
import LBB.Design.png as PNG

# Specify output folders
resources_folder = f"{Config.course_root}/_resources/designs"
output_folder = f"{resources_folder}/layout"
svg_folder = f"{output_folder}/svg"
png_folder = f"{output_folder}/png"
animated_folder = f"{output_folder}/animated"

# Clear/create folders
Utilities.clear_folder(svg_folder)
Utilities.clear_folder(png_folder)
Utilities.clear_folder(animated_folder)

# ---
# LBB
# ---
layout = Layout.Layout("LBB", 4, 7, Config.box_names, 13.0, 0.125, 1.25, "#000000", "#FFFFFF", True, True)
svg = SVG.SVG("layout_LBB", None, 98.75, 56.0, "0 0 98.75 56.0", layout.boxes, _with_profile=False, _with_title=False, _with_labels=True)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, dpi=600).convert(svg_path, png_path)

# --------
# Bootcamp
# --------
layout = Layout.Layout("Bootcamp", 3, 7, Config.bootcamp_box_names, 13.0, 0.125, 1.25, "#000000", "#FFFFFF", True, True)
svg = SVG.SVG("layout_bootcamp", None, 98.75, 56.0, "0 0 98.75 56.0", layout.boxes, _with_profile=False, _with_title=False, _with_labels=True)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, dpi=600).convert(svg_path, png_path)

# -------------
# Build a Brain
# -------------
layout = Layout.Layout("Build a Brain", 2, 4, Config.buildabrain_box_names, 13.0, 0.125, 1.25, "#000000", "#FFFFFF", True, True)
svg = SVG.SVG("layout_buildabrain", None, 98.75, 56.0, "0 0 98.75 56.0", layout.boxes, _with_profile=False, _with_title=False, _with_labels=True)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, dpi=600).convert(svg_path, png_path)

# ---------
# Own Phone
# ---------
layout = Layout.Layout("Own Phone", 3, 3, Config.ownphone_box_names, 13.0, 0.125, 1.25, "#000000", "#FFFFFF", True, True)
svg = SVG.SVG("layout_ownphone", None, 98.75, 56.0, "0 0 98.75 56.0", layout.boxes, _with_profile=False, _with_title=False, _with_labels=True)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, dpi=600).convert(svg_path, png_path)

# ----------------
# Animated (hover)
# ----------------
logo = Logo.Logo("LBB", 0.0, 0.0, Config.box_names, Logo.LBB_box_parameters, 13.0, 0.5, _with_labels=True)
animation_parameters_path = f"{output_folder}/animation_parameters_position_hover.csv"
svg = SVG.SVG("layout_animated_hover", None, 100, 100, "0 0 100 100", logo.boxes, _with_profile=False, _with_title=False, _with_labels=True)
svg_path = f"{output_folder}/animated/{svg.name}.svg"
svg.animate(animation_parameters_path, True, False, True, svg_path)

#FIN