# Generate Video Thumbnails

# Import Libraries
import os
import glob

# Import Modules
import Design.png as PNG

# Specify paths
repo_path = "/home/kampff/NoBlackBoxes/LastBlackBox"
boxes_path = repo_path + "/boxes"
videos_path = repo_path + "/course/_videos"

# Read templates
LBB_template_path = videos_path + "/_templates/LBB_title.svg"
with open(LBB_template_path, 'r') as file:
    LBB_template_text = file.read()
NB3_template_path = videos_path + "/_templates/NB3_title.svg"
with open(NB3_template_path, 'r') as file:
    NB3_template_text = file.read()

# Find all boxes at "boxes_path"
box_folders = [f for f in glob.glob(boxes_path + '/*') if os.path.isdir(f)]

# Process each video's *.md file 
for box_folder in box_folders:
    # Extract (and capitalize) box name
    box_name = os.path.basename(box_folder)
    box_name = box_name[0].upper() + box_name[1:]

    # Find all *.md files in each box
    md_files = glob.glob(box_folder + '/*.md')
    for md_file in md_files:
        md_basename = os.path.basename(md_file)
        # Is an NB3 video?
        if md_basename[:4] == "NB3_":
            is_NB3 = True
            video_name = md_basename[4:-3].replace('-', ' ')
            svg_text = NB3_template_text
        else:
            is_NB3 = False
            video_name = md_basename[:-3].replace('-', ' ')
            svg_text = LBB_template_text
        # Alter SVG template
        svg_text = svg_text.replace(">Box<", f">{box_name}<")
        svg_text = svg_text.replace(">Title of the Video<", f">{video_name}<")
        svg_path = md_file[:-3] + ".svg"
        png_path = svg_path[:-4] + ".png"
        with open(svg_path, 'w') as file:
                    num_bytes = file.write(svg_text)
        PNG.PNG(video_name, dpi=96, page=True).convert(svg_path, png_path)
        print(f"Created SVG and PNG for {box_name}:{video_name}")

#FIN