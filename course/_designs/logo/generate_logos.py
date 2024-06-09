# Generate LBB logos

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
import Design.box as Box
import Design.profile as Profile
import Design.text as Text

# Specify output folders
base_path = "/home/kampff/NoBlackBoxes/LastBlackBox/course/_designs"
output_folder = f"{base_path}/logo"
svg_folder = f"{output_folder}/svg"
png_folder = f"{output_folder}/png"
icons_folder = f"{output_folder}/icons"
animated_folder = f"{output_folder}/animated"

# Clear/create folders
Utilities.clear_folder(svg_folder)
Utilities.clear_folder(png_folder)
Utilities.clear_folder(animated_folder)
Utilities.clear_folder(icons_folder)

# ---
# LBB
# ---
box_parameters_path = f"{output_folder}/box_parameters_LBB.csv"
# No Text
svg = SVG.SVG("logo_LBB", None, 100, 100, "0 0 100 100", with_profile=False, with_title=False)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(box_parameters_path, svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, width=1000, height=1000).convert(svg_path, png_path)
# White Text
title = Text.Text("LBB", "The Last Black Box", 8.5, 87.5, 1.25, 'FFFFFF', 9.0, 'Arial', "")
svg = SVG.SVG("logo_LBB_white_text", title, 100, 100, "0 0 100 100", with_profile=False, with_title=True)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(box_parameters_path, svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, width=1000, height=1000).convert(svg_path, png_path)
# Black Text
title = Text.Text("LBB", "The Last Black Box", 8.5, 87.5, 1.25, '000000', 9.0, 'Arial', "")
svg = SVG.SVG("logo_LBB_black_text", title, 100, 100, "0 0 100 100", with_profile=False, with_title=True)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(box_parameters_path, svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, width=1000, height=1000).convert(svg_path, png_path)

# ---
# NBB
# ---
box_parameters_path = f"{output_folder}/box_parameters_NBB.csv"
# No Text
svg = SVG.SVG("logo_NBB", None, 100, 100, "0 0 100 100", with_profile=False, with_title=False)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(box_parameters_path, svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, width=1000, height=1000).convert(svg_path, png_path)
# White Text
title = Text.Text("NBB", "No Black Boxes", 14.0, 86.5, 1.25, 'FFFFFF', 9.5, 'Arial', "")
svg = SVG.SVG("logo_NBB_white_text", title, 100, 100, "0 0 100 100", with_profile=False, with_title=True)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(box_parameters_path, svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, width=1000, height=1000).convert(svg_path, png_path)
# Black Text
title = Text.Text("NBB", "No Black Boxes", 14.0, 86.5, 1.25, '000000', 9.5, 'Arial', "")
svg = SVG.SVG("logo_NBB_black_text", title, 100, 100, "0 0 100 100", with_profile=False, with_title=True)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(box_parameters_path, svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, width=1000, height=1000).convert(svg_path, png_path)
# Profile
svg = SVG.SVG("logo_NBB_profile", None, 100, 125, "0 0 100 125", with_profile=True, with_title=False)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(box_parameters_path, svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, width=1000, height=1250).convert(svg_path, png_path)

# ---------
# NBB Icons
# ---------
box_parameters_path = f"{output_folder}/box_parameters_NBB.csv"
svg = SVG.SVG("icon_NBB", None, 100, 100, "9.5 10 80 80", with_profile=False, with_title=False)
svg_path = f"{output_folder}/icons/{svg.name}.svg"
svg.draw(box_parameters_path, svg_path)
png_path = f"{output_folder}/icons/{svg.name}_1024.png"
PNG.PNG(svg.name, width=1024, height=1024).convert(svg_path, png_path)
png_path = f"{output_folder}/icons/{svg.name}_512.png"
PNG.PNG(svg.name, width=512, height=512).convert(svg_path, png_path)
png_path = f"{output_folder}/icons/{svg.name}_256.png"
PNG.PNG(svg.name, width=256, height=256).convert(svg_path, png_path)
png_path = f"{output_folder}/icons/{svg.name}_128.png"
PNG.PNG(svg.name, width=128, height=128).convert(svg_path, png_path)
png_path = f"{output_folder}/icons/{svg.name}_64.png"
PNG.PNG(svg.name, width=64, height=64).convert(svg_path, png_path)

# ---------
# LBB Icons
# ---------
box_parameters_path = f"{output_folder}/box_parameters_LBB.csv"
svg = SVG.SVG("icon_LBB", None, 100, 100, "9.5 10 80 80", with_profile=False, with_title=False)
svg_path = f"{output_folder}/icons/{svg.name}.svg"
svg.draw(box_parameters_path, svg_path)
png_path = f"{output_folder}/icons/{svg.name}_1024.png"
PNG.PNG(svg.name, width=1024, height=1024).convert(svg_path, png_path)
png_path = f"{output_folder}/icons/{svg.name}_512.png"
PNG.PNG(svg.name, width=512, height=512).convert(svg_path, png_path)
png_path = f"{output_folder}/icons/{svg.name}_256.png"
PNG.PNG(svg.name, width=256, height=256).convert(svg_path, png_path)
png_path = f"{output_folder}/icons/{svg.name}_128.png"
PNG.PNG(svg.name, width=128, height=128).convert(svg_path, png_path)
png_path = f"{output_folder}/icons/{svg.name}_64.png"
PNG.PNG(svg.name, width=64, height=64).convert(svg_path, png_path)

# --------
# Animated
# --------
box_parameters_path = f"{output_folder}/box_parameters_NBB.csv"
# NBB
animation_parameters_path = f"{output_folder}/animation_parameters_fill_stroke_delay.csv"
svg = SVG.SVG("logo_NBB_animated_delay", None, 100, 100, "0 0 100 100", with_profile=False, with_title=False)
svg_path = f"{output_folder}/animated/{svg.name}.svg"
svg.animate(box_parameters_path, animation_parameters_path, False, False, svg_path)
# NBB (hover)
animation_parameters_path = f"{output_folder}/animation_parameters_fill_stroke_hover.csv"
svg = SVG.SVG("logo_NBB_animated_hover", None, 100, 100, "0 0 100 100", with_profile=False, with_title=False)
svg_path = f"{output_folder}/animated/{svg.name}.svg"
svg.animate(box_parameters_path, animation_parameters_path, True, False, svg_path)
# NBB (repeat)
animation_parameters_path = f"{output_folder}/animation_parameters_fill_stroke_repeat.csv"
svg = SVG.SVG("logo_NBB_animated_repeat", None, 100, 100, "0 0 100 100", with_profile=False, with_title=False)
svg_path = f"{output_folder}/animated/{svg.name}.svg"
svg.animate(box_parameters_path, animation_parameters_path, False, True, svg_path)
# Profile
animation_parameters_path = f"{output_folder}/animation_parameters_fill_stroke_delay.csv"
svg = SVG.SVG("logo_NBB_profile_animated_delay", None, 100, 125, "0 0 100 125", with_profile=True, with_title=False)
svg_path = f"{output_folder}/animated/{svg.name}.svg"
svg.animate(box_parameters_path, animation_parameters_path, False, False, svg_path)
# Profile (hover)
animation_parameters_path = f"{output_folder}/animation_parameters_fill_stroke_hover.csv"
svg = SVG.SVG("logo_NBB_profile_animated_hover", None, 100, 125, "0 0 100 125", with_profile=True, with_title=False)
svg_path = f"{output_folder}/animated/{svg.name}.svg"
svg.animate(box_parameters_path, animation_parameters_path, True, False, svg_path)
# Profile (repeat)
animation_parameters_path = f"{output_folder}/animation_parameters_fill_stroke_repeat.csv"
svg = SVG.SVG("logo_NBB_profile_animated_repeat", None, 100, 125, "0 0 100 125", with_profile=True, with_title=False)
svg_path = f"{output_folder}/animated/{svg.name}.svg"
svg.animate(box_parameters_path, animation_parameters_path, False, True, svg_path)

#FIN