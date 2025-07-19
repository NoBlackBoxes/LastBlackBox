# Generate Video GIFs

# Import Libraries
import os
import glob
import cv2
from PIL import Image
import numpy as np
import LBB.config as Config

# Find all boxes at "boxes_path"
box_folders = [f for f in glob.glob(Config.boxes_root + '/*') if os.path.isdir(f)]

# Process each video's *.md file 
for box_name in Config.box_names:

    # Determine box folder
    box_folder = f"{Config.boxes_root}/{box_name.lower()}"

    # Specify folders
    lessons_folder = box_folder + "/_resources/lessons"
    thumbnails_folder = lessons_folder + "/thumbnails"
    video_folder = f"/home/kampff/Dropbox/Voight-Kampff/NoBlackBoxes/LastBlackBox/videos/boxes/{box_name.lower()}"

    # Find all *.md files in each box's lessons folder
    md_files = glob.glob(lessons_folder + '/*.md')
    for md_file in md_files:
        md_basename = os.path.basename(md_file)

        # Extract lesson type and name
        with open(md_file) as f:
            first_line = f.readline()
        lesson_header = first_line.split(':')
        if lesson_header[1].strip() == 'NB3':
            is_NB3 = True
            lesson_name = lesson_header[2].strip()
        else:
            is_NB3 = False
            lesson_name = lesson_header[1].strip()

        # Extract video name
        video_path = f"{video_folder}/{md_basename[:-3]}.mkv"
        print(f"Video Name: {video_path}")

        # Extract thumbnail name
        png_path = f"{thumbnails_folder}/{md_basename[:-3]}.png"
        print(f"PNG Name: {png_path}")

        # Specify GIF output path
        gif_path = f"{thumbnails_folder}/{md_basename[:-3]}.gif"
        print(f"GIF Name: {gif_path}")

        # Load thumbnail

        # Load video
        video = cv2.VideoCapture(video_path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, num_frames - 100, num=30, dtype=int)
        frames = []
        target_size = (320, 180)
        for idx in frame_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = video.read()
            if success:
                resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
            else:
                print(f"Failed to read frame at index {idx}")
        video.release()

        # Generate GIF
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=300, loop=0, optimize=True)

        # DEBUG
        exit()

#FIN