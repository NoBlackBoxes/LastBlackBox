# Generate LBB logos

# Imports
import LBB.utilities as Utilities
import LBB.config as Config
import LBB.Design.logo as Logo
import LBB.Design.svg as SVG
import LBB.Design.png as PNG
import LBB.Design.text as Text

# Specify paths
resources_folder = f"{Config.course_root}/_resources/designs"
output_folder = f"{resources_folder}/logo"
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
logo = Logo.Logo("LBB", 0.0, 0.0, Config.box_names, Logo.LBB_box_parameters, 13.0, 0.5)
# No Text
svg = SVG.SVG("logo_LBB", None, 100, 100, "0 0 100 100", logo.boxes, _with_profile=False, _with_title=False)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, width=1000, height=1000).convert(svg_path, png_path)
# White Text
title = Text.Text("LBB", "The Last Black Box", 8.5, 87.5, 1.25, 'FFFFFF', 9.0, 'Arial', "")
svg = SVG.SVG("logo_LBB_white_text", title, 100, 100, "0 0 100 100", logo.boxes, _with_profile=False, _with_title=True)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, width=1000, height=1000).convert(svg_path, png_path)
# Black Text
title = Text.Text("LBB", "The Last Black Box", 8.5, 87.5, 1.25, '000000', 9.0, 'Arial', "")
svg = SVG.SVG("logo_LBB_black_text", title, 100, 100, "0 0 100 100", logo.boxes, _with_profile=False, _with_title=True)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, width=1000, height=1000).convert(svg_path, png_path)

# ---
# NBB
# ---
logo = Logo.Logo("NBB", 0.0, 0.0, Config.box_names, Logo.NBB_box_parameters, 13.0, 0.5)
# No Text
svg = SVG.SVG("logo_NBB", None, 100, 100, "0 0 100 100", logo.boxes, _with_profile=False, _with_title=False)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, width=1000, height=1000).convert(svg_path, png_path)
# White Text
title = Text.Text("NBB", "No Black Boxes", 14.0, 86.5, 1.25, 'FFFFFF', 9.5, 'Arial', "")
svg = SVG.SVG("logo_NBB_white_text", title, 100, 100, "0 0 100 100", logo.boxes, _with_profile=False, _with_title=True)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, width=1000, height=1000).convert(svg_path, png_path)
# Black Text
title = Text.Text("NBB", "No Black Boxes", 14.0, 86.5, 1.25, '000000', 9.5, 'Arial', "")
svg = SVG.SVG("logo_NBB_black_text", title, 100, 100, "0 0 100 100", logo.boxes, _with_profile=False, _with_title=True)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, width=1000, height=1000).convert(svg_path, png_path)
# Profile
svg = SVG.SVG("logo_NBB_profile", None, 100, 125, "0 0 100 125", logo.boxes, _with_profile=True, _with_title=False)
svg_path = f"{output_folder}/svg/{svg.name}.svg"
svg.draw(svg_path)
png_path = f"{output_folder}/png/{svg.name}.png"
PNG.PNG(svg.name, width=1000, height=1250).convert(svg_path, png_path)

# ---------
# LBB Icons
# ---------
logo = Logo.Logo("LBB", 0.0, 0.0, Config.box_names, Logo.LBB_box_parameters, 13.0, 0.5)
svg = SVG.SVG("icon_LBB", None, 100, 100, "9.5 10 80 80", logo.boxes, _with_profile=False, _with_title=False)
svg_path = f"{output_folder}/icons/{svg.name}.svg"
svg.draw(svg_path)
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
# NBB Icons
# ---------
logo = Logo.Logo("NBB", 0.0, 0.0, Config.box_names, Logo.NBB_box_parameters, 13.0, 0.5)
svg = SVG.SVG("icon_NBB", None, 100, 100, "9.5 10 80 80", logo.boxes, _with_profile=False, _with_title=False)
svg_path = f"{output_folder}/icons/{svg.name}.svg"
svg.draw(svg_path)
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
# NBB
animation_parameters_path = f"{output_folder}/animation_parameters_fill_stroke_delay.csv"
svg = SVG.SVG("logo_NBB_animated_delay", None, 100, 100, "0 0 100 100", logo.boxes, _with_profile=False, _with_title=False)
svg_path = f"{output_folder}/animated/{svg.name}.svg"
svg.animate(animation_parameters_path, False, False, False, svg_path)
# NBB (hover)
animation_parameters_path = f"{output_folder}/animation_parameters_fill_stroke_hover.csv"
svg = SVG.SVG("logo_NBB_animated_hover", None, 100, 100, "0 0 100 100", logo.boxes, _with_profile=False, _with_title=False)
svg_path = f"{output_folder}/animated/{svg.name}.svg"
svg.animate(animation_parameters_path, True, False, False, svg_path)
# NBB (repeat)
animation_parameters_path = f"{output_folder}/animation_parameters_fill_stroke_repeat.csv"
svg = SVG.SVG("logo_NBB_animated_repeat", None, 100, 100, "0 0 100 100", logo.boxes, _with_profile=False, _with_title=False)
svg_path = f"{output_folder}/animated/{svg.name}.svg"
svg.animate(animation_parameters_path, False, True, False, svg_path)
# Profile
animation_parameters_path = f"{output_folder}/animation_parameters_fill_stroke_delay.csv"
svg = SVG.SVG("logo_NBB_profile_animated_delay", None, 100, 125, "0 0 100 125", logo.boxes, _with_profile=True, _with_title=False)
svg_path = f"{output_folder}/animated/{svg.name}.svg"
svg.animate(animation_parameters_path, False, False, False, svg_path)
# Profile (hover)
animation_parameters_path = f"{output_folder}/animation_parameters_fill_stroke_hover.csv"
svg = SVG.SVG("logo_NBB_profile_animated_hover", None, 100, 125, "0 0 100 125", logo.boxes, _with_profile=True, _with_title=False)
svg_path = f"{output_folder}/animated/{svg.name}.svg"
svg.animate(animation_parameters_path, True, False, False, svg_path)
# Profile (repeat)
animation_parameters_path = f"{output_folder}/animation_parameters_fill_stroke_repeat.csv"
svg = SVG.SVG("logo_NBB_profile_animated_repeat", None, 100, 125, "0 0 100 125", logo.boxes, _with_profile=True, _with_title=False)
svg_path = f"{output_folder}/animated/{svg.name}.svg"
svg.animate(animation_parameters_path, False, True, False, svg_path)

#FIN