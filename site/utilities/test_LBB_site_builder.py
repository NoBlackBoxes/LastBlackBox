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
import LBB.instruction as Instruction
import LBB.image as Image
import LBB.video as Video
import LBB.task as Task

# Reload modules
import importlib
importlib.reload(Box)
importlib.reload(Topic)
importlib.reload(Lesson)
importlib.reload(Instruction)
importlib.reload(Image)
importlib.reload(Video)
importlib.reload(Task)
#----------------------------------------------------------

readme_path = "/home/kampff/NoBlackBoxes/LastBlackBox/boxes/electrons/README_new.md"
box = Box.Box(readme_path)

output = box.topics[0].render()
with open("output.html", "w") as file:
    file.write(output)

# To Do
# - parse video (new class)
# - parse task (class)
# - parse images
# - create task spec (box/button labels...)
# - Think about task completion assement (automated)

# Print Box
print(box.name)
print("-------------")
print(box.description[0])
print("-------------")
for topic in box.topics:
    print("\t" + topic.name)
    print("\t:: " + topic.description[0])
    print("\t-------------")
    for lesson in topic.lessons:
        print("\t" + str(lesson.level))
        for step in lesson.steps:
            print(f"\t\t{step}")
print("-------------")
