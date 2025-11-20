# -*- coding: utf-8 -*-
"""
Generate LBB Courses (session READMEs) from templates

@author: kampff
"""

# Imports
import LBB.config as Config
import LBB.utilities as Utilities
import LBB.Engine.course as Course

# Reload libraries
import importlib
importlib.reload(Utilities)
importlib.reload(Config)
importlib.reload(Course)

# Build courses
for course_name in Config.course_names:
    print(f"Building \"{course_name}\"...")
    course = Course.Course(course_name)

    # Render README for each session
    for session in course.sessions:
        session_folder = f"{Config.course_path}/versions/{course.slug}/{(session.index+1):02d}_{session.slug}"
        README_text = session.render(course_name, type="MD")
        README_path = f"{session_folder}/README.md"
        with open(README_path, 'w', encoding='utf8') as f:
            f.writelines(README_text)    

#FIN