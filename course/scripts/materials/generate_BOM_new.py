# -*- coding: utf-8 -*-
"""
Generate LBB Bill of Materials (BOM)

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
box_names = ['Atoms']

# Load each box
for box_count, box_name in enumerate(box_names):

    # Determine box folder
    box_folder = f"{Config.boxes_root}/{box_name.lower()}"

    # Specify lessons folder
    lessons_folder = box_folder + "/_lessons"

    # Find all lessons (*.md files) in each box's lessons folder
    lesson_paths = glob.glob(lessons_folder + '/*.md')
    print(f"{box_name}")
    for lesson_path in lesson_paths:

        # Extract lesson slug
        lesson_slug = os.path.basename(lesson_path)
        print(f"- {lesson_slug}")

# FIN