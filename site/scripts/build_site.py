# -*- coding: utf-8 -*-
"""
Build the LBB site

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
#----------------------------------------------------------

# Import libraries
import os
import LBB.utilities as Utilities

# Import modules
import LBB.course as Course
import LBB.session as Session
import LBB.topic as Topic
import LBB.lesson as Lesson
import LBB.instruction as Instruction
import LBB.image as Image
import LBB.video as Video
import LBB.task as Task
import LBB.input as Input

# Reload libraies and modules
import importlib
importlib.reload(Utilities)
importlib.reload(Course)
importlib.reload(Session)
importlib.reload(Topic)
importlib.reload(Lesson)
importlib.reload(Instruction)
importlib.reload(Image)
importlib.reload(Video)
importlib.reload(Task)
importlib.reload(Input)

#----------------------------------------------------------

# List all *available* courses
repo_root = "/home/kampff/NoBlackBoxes/LastBlackBox"
course_root = repo_root + "/course"
#course_names = ["bootcamp", "buildabrain"]
course_names = ["buildabrain"]

# For each course...
for course_name in course_names:
    course_folder = course_root + "/" + course_name
    course = Course.Course(course_folder)

    # Find all sessions in the course

# Copy media from Repo (to static)

readme_path = "/home/kampff/NoBlackBoxes/LastBlackBox/boxes/electrons/README_new.md"
session = Session.Session(readme_path)

# Create output folder
output_path = '/home/kampff/NoBlackBoxes/LastBlackBox/site/templates'

# Render box
box.render_topics(output_path)

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
