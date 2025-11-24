# Generate Video Thumbnails
# - NOTE: Requires that the Arial and Verdana "Windows Fonts" are installed for use by Inkscape

# Import Libraries
import os
import glob
import LBB.config as Config
import LBB.utilities as Utilities
import LBB.Design.png as PNG

# Specify paths
resources_folder = f"{Config.course_path}/_resources/videos"

# Specify overwrite (re-do thumbnail generation)
overwrite = False

# Read templates
LBB_template_path = resources_folder + "/templates/LBB_title.svg"
with open(LBB_template_path, 'r') as file:
    LBB_template_text = file.read()
NB3_template_path = resources_folder + "/templates/NB3_title.svg"
with open(NB3_template_path, 'r') as file:
    NB3_template_text = file.read()

# Find all boxes at "boxes_path"
box_folders = [f for f in glob.glob(Config.boxes_path + '/*') if os.path.isdir(f)]

# Process each video's *.md file 
for box_name in Config.box_names:

    # Determine box folder
    box_folder = f"{Config.boxes_path}/{box_name.lower()}"

    # Specify lessons folder
    lessons_folder = box_folder + "/_resources/lessons"

    # Confirm thumbnails folder
    thumbnails_folder = lessons_folder + "/thumbnails"
    Utilities.confirm_folder(thumbnails_folder)

    # Find all *.md files in each box's lessons folder
    md_files = glob.glob(lessons_folder + '/*.md')
    for md_file in md_files:
        md_basename = os.path.basename(md_file)

        # Check for existing thumbnail (if it exists and not overwriting, skip)
        thumbnail_path = f"{thumbnails_folder}/{md_basename[:-3]}.svg"
        if not overwrite and os.path.exists(thumbnail_path):
            continue

        # Extract lesson type and name
        with open(md_file) as f:
            first_line = f.readline()
        lesson_header = first_line.split(':')
        if lesson_header[1].strip() == 'NB3':
            is_NB3 = True
            lesson_name = lesson_header[2].strip()
            svg_text = NB3_template_text
        else:
            is_NB3 = False
            lesson_name = lesson_header[1].strip()
            svg_text = LBB_template_text

        # Alter SVG template
        svg_text = svg_text.replace(">Box<", f">{box_name}<")
        svg_text = svg_text.replace(">Title of the Video<", f">{lesson_name}<")
        svg_path = f"{thumbnails_folder}/{md_basename[:-3]}.svg"
        png_path = svg_path[:-4] + ".png"
        with open(svg_path, 'w') as file:
                    num_bytes = file.write(svg_text)
        PNG.PNG(lesson_name, dpi=96, page=True).convert(svg_path, png_path)
        print(f"Created SVG and PNG for {box_name}:{lesson_name}")

#FIN