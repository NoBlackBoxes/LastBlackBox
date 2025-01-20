# -*- coding: utf-8 -*-
"""
Generate LBB packaging CAD files

@author: kampff
"""

# Import Libraries
import os
import glob
import re
import cadquery as cq

# Import modules
import LBB.Engine.config as Config
import LBB.Engine.utilities as Utilities
import LBB.Engine.course as Course
import LBB.Engine.box as Box

# Reload libraries and modules
import importlib
importlib.reload(Config)
importlib.reload(Utilities)
importlib.reload(Course)
importlib.reload(Box)

# Build CAD packaging for each course
course_names = ["The Last Black Box", "Bootcamp", "Build a Brain"]
for course_name in course_names:
    course = Course.Course(course_name)

    # Generate CAD object
    CAD_folder = f"{Config.course_root}/_resources/materials/CAD/{course.slug}"
    Utilities.clear_folder(CAD_folder)
    for session in course.sessions:
        for box in session.boxes:
            for m in box.materials:
                output_path = f"{CAD_folder}/{m.slug}.step"
                cuboid = cq.Workplane("XY").box(m.x, m.y, m.x)
                cuboid.val().label = m.slug
                cq.exporters.export(cuboid, output_path, exportType=cq.exporters.ExportTypes.STEP)
                with open(output_path, 'r') as file:
                   content = file.read()
                content = content.replace("Open CASCADE STEP translator ", m.slug)
                with open(output_path, 'w') as file:
                    file.write(content)

                
# FIN