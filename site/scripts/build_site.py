# -*- coding: utf-8 -*-
"""
Build the LBB site

@author: kampff
"""

# Imports
import LBB.config as Config
import LBB.utilities as Utilities
import LBB.Engine.course as Course

# Reload libraries and modules
import importlib
importlib.reload(Config)
importlib.reload(Utilities)
importlib.reload(Course)

# Set paths
templates_folder = Config.site_root + "/templates"
courses_templates_folder = templates_folder + "/courses"

# Load and render each course
for course_name in Config.course_names:

    # Load course
    print(f"Loading \"{course_name}\"...")
    course = Course.Course(course_name)
    course_folder = f"{courses_templates_folder}/{course.slug}"
    Utilities.clear_folder(course_folder)

    # Render HTML for each lesson
    for session in course.sessions:
        session_folder = f"{course_folder}/{session.slug}"
        Utilities.clear_folder(session_folder)
        for box in session.boxes:
            box_folder = f"{session_folder}/{box.slug}"
            Utilities.clear_folder(box_folder)
            for lesson in box.lessons:
                HTML_text = lesson.render(type="MD")
                HTML_path = f"{box_folder}/{lesson.slug}.html"
                with open(HTML_path, 'w', encoding='utf8') as f:
                    f.writelines(HTML_text)    

# FIN