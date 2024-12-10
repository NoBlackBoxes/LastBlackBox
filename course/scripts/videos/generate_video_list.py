# Generate Video List
"""
Generate list of all LBB videos

@author: kampff
"""

# Import Libraries
import os
import glob

# Import Modules
import LBB.config as Config

# Specify paths
video_list_path = Config.course_root + "/_resources/videos/list.md"

# Open video list file
video_list_file = open(video_list_path, 'w')

# Find all videos in each box's lessons
completed = 0
todo = 0
for box_name in Config.box_names:

    # Determine box folder
    box_folder = f"{Config.boxes_root}/{box_name.lower()}"

    # Write box header to video list
    video_list_file.write(f"## {box_name}\n")

    # Specify lessons folder
    lessons_folder = box_folder + "/_lessons"

    # Find all lessons (*.md files) in each box's lessons folder
    lesson_paths = glob.glob(lessons_folder + '/*.md')
    for lesson_path in lesson_paths:

        # Extract lesson slug
        lesson_slug = os.path.basename(lesson_path)[:-3]

        # Read lesson text
        with open(lesson_path, encoding='utf8') as f:
            lines = f.readlines()

        # Extract lesson type and name
        lesson_header = lines[0].split(':')
        if lesson_header[1].strip() == 'NB3':
            is_NB3 = True
            lesson_name = lesson_header[2].strip()
        else:
            is_NB3 = False
            lesson_name = lesson_header[1].strip()

        # Extract video URL
        max_count = len(lines)
        line_count = 1
        while line_count < max_count:
            line = lines[line_count]
            if line.startswith("## [Video]"):
                video_URL = line.split('(')[1].split(')')[0]
                if video_URL:
                    if is_NB3:
                        video_list_file.write(f"- [x] [NB3 : {lesson_name}]({video_URL})\n")
                    else:
                        video_list_file.write(f"- [x] [{lesson_name}]({video_URL})\n")
                    completed += 1
                else:
                    if is_NB3:
                        video_list_file.write(f"- [ ] [NB3 : {lesson_name}]()\n")
                    else:
                        video_list_file.write(f"- [ ] [{lesson_name}]()\n")
                    todo += 1
                line_count = max_count
            line_count += 1

    # Write spacer to video list
    video_list_file.write(f"\n")

# Close video list file
video_list_file.close()

# Report
print(f"{completed} videos completed!\n{todo} videos to do.")

#FIN