# -*- coding: utf-8 -*-
"""
Generate LBB "README.md" files

@author: kampff
"""

# Import Libraries
import os
import glob

# Import modules
import LBB.config as Config
import LBB.utilities as Utilities
import LBB.box as Box

# Reload libraries and modules
import importlib
importlib.reload(Config)
importlib.reload(Utilities)
importlib.reload(Box)

# List all boxes
box_names = Config.box_names

## DEBUG
box_names = ['Electrons','Atoms']

##  WORK IN PROGRESS
# Load each box's template
# Render lessons and projects

# Load each box's lessons
for box_count, box_name in enumerate(box_names):

    # Determine box folder
    box_folder = f"{Config.boxes_root}/{box_name.lower()}"

    # Specify lessons folder
    lessons_folder = box_folder + "/_lessons"

    # Find all lessons (*.md files) in each box's lessons folder
    lesson_paths = glob.glob(lessons_folder + '/*.md')
    for lesson_path in lesson_paths:

        # Extract lesson slug
        lesson_slug = os.path.basename(lesson_path)

    # Load "lessons.md"
    lessons_path = box_folder + "/lessons.md"
    with open(lessons_path, encoding='utf8') as f:
        lines = f.readlines()
    text = []
    for line in lines:
        if line.strip():                # Remove empty lines
            text.append(line.rstrip())  # Remove trailing whitespace (including '/n')

    # Build box
    box = Box.Box(text=text)
    box.index = box_count

    # Store box
    box_data_path = f"{box_folder}/lessons.json"
    box.store(box_data_path)

# FIN