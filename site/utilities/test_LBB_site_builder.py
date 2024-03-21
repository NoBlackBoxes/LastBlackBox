# -*- coding: utf-8 -*-
"""
Test the LBB site building tools

@author: kampff
"""
#----------------------------------------------------------
# Load environment file and variables
import os
from dotenv import load_dotenv
load_dotenv()
libs_path = os.getenv('LIBS_PATH')
base_path = os.getenv('BASE_PATH')

# Set library paths
import sys
sys.path.append(libs_path)

# Import libraries
import os
import numpy as np

# Import local modules
import LBB.box as Box
import LBB.topic as Topic
import LBB.lesson as Lesson

# Reload modules
import importlib
importlib.reload(Box)
importlib.reload(Topic)
importlib.reload(Lesson)
#----------------------------------------------------------

readme_path = "/home/kampff/NoBlackBoxes/LastBlackBox/boxes/electrons/README_new.md"
box = Box.Box(readme_path)

# Print Box
print(box.name)
print("-------------")
print(box.description[0])
print("-------------")
for topic in box.topics:
    print("\t" + topic.name)
    print("\t  " + topic.description[0])
    print("\t-------------")
    for lesson in topic.lessons:
        print("\t" + str(lesson.level))
        print("\t\t" + lesson.instructions[0])
print("-------------")
