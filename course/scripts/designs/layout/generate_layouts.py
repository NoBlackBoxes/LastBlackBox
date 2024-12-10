# Generate Course Layouts

# Import Libraries
import Design.utilities as Utilities

# Import Modules
import LBB.config as Config
import Design.svg as SVG
import Design.png as PNG

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
box_parameters_path = f"{output_folder}/box_parameters_LBB.csv"
svg = SVG.SVG("layout_LBB", None, 98.75, 56.0, "0 0 98.75 56.0", with_profile=False, with_title=False, with_labels=True)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(box_parameters_path, svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, dpi=600).convert(svg_path, png_path)

# --------
# Bootcamp
# --------
box_parameters_path = f"{output_folder}/box_parameters_bootcamp.csv"
svg = SVG.SVG("layout_bootcamp", None, 98.75, 56.0, "0 0 98.75 56.0", with_profile=False, with_title=False, with_labels=True)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(box_parameters_path, svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, dpi=600).convert(svg_path, png_path)

# -------------
# Build a Brain
# -------------
box_parameters_path = f"{output_folder}/box_parameters_buildabrain.csv"
svg = SVG.SVG("layout_buildabrain", None, 98.75, 56.0, "0 0 98.75 56.0", with_profile=False, with_title=False, with_labels=True)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(box_parameters_path, svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, dpi=600).convert(svg_path, png_path)

# ---------
# Own Phone
# ---------
box_parameters_path = f"{output_folder}/box_parameters_ownphone.csv"
svg = SVG.SVG("layout_ownphone", None, 98.75, 56.0, "0 0 98.75 56.0", with_profile=False, with_title=False, with_labels=True)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(box_parameters_path, svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, dpi=600).convert(svg_path, png_path)

# ----------------
# Animated (hover)
# ----------------
box_parameters_path = f"{Config.course_root}/_resources/designs/logo/box_parameters_LBB.csv"
animation_parameters_path = f"{output_folder}/animation_parameters_position_hover.csv"
svg = SVG.SVG("layout_animated_hover", None, 100, 100, "0 0 100 100", with_profile=False, with_title=False, with_labels=True)
svg_path = f"{output_folder}/animated/{svg.name}.svg"
svg.animate(box_parameters_path, animation_parameters_path, True, False, True, svg_path)

#FIN