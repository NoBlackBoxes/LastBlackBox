# -*- coding: utf-8 -*-
"""
Generate LBB Courses (session READMEs) from templates

@author: kampff
"""

# Imports
import LBB.config as Config
import LBB.utilities as Utilities
import LBB.Engine.course as Course

# Build courses
for course_name in Config.course_names:
    print(f"Building \"{course_name}\"...")
    course = Course.Course(course_name)

    # Render README for each session
    for session in course.sessions:
        session_folder = f"{Config.course_path}/versions/{course.slug}/{(session.index+1):02d}_{session.slug}"
        README_text = session.render()
        README_path = f"{session_folder}/README.md"
        with open(README_path, 'w', encoding='utf8') as f:
            f.writelines(README_text)

## Build LBB box READMEs
#course_name in Config.course_names:
#    print(f"Building \"{course_name}\"...")
#    course = Course.Course(course_name)
#
#    # Render README for each session
#    for session in course.sessions:
#        session_folder = f"{Config.course_path}/versions/{course.slug}/{(session.index+1):02d}_{session.slug}"
#        README_text = session.render()
#        README_path = f"{session_folder}/README.md"
#        with open(README_path, 'w', encoding='utf8') as f:
#            f.writelines(README_text)
#
#
#        # Render README for each "box" in the LBB repo (same as "full" course README)
#        if course_name == "The Last Black Box":
#            README_path = f"{Config.boxes_path}/{session.slug}/README.md"
#            Config.image_prefix = "../.."   # Adjust image prefix
#            with open(README_path, 'w', encoding='utf8') as f:
#                f.writelines(README_text)
#            image_prefix = "../../../.."    # Reset image prefix


#FIN