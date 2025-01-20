# -*- coding: utf-8 -*-
"""
Generate LBB Bill of Materials (BOM)

@author: kampff
"""

# Import Libraries
import os
import glob

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

# Build courses
course_names = ["The Last Black Box", "Bootcamp", "Build a Brain"]
for course_name in course_names:
    course = Course.Course(course_name)

    # Generate BOM
    BOM_path = f"{Config.course_root}/_resources/materials/BOM/{course.slug}_BOM.csv"
    with open(BOM_path, 'w') as file:
        file.write("Name,Depth,Description,Quantity,Datasheet,Supplier,Package,x(mm),y(mm),z(mm)\n")
        file.write(",,,,,,,,,\n")
        for session in course.sessions:
            for box in session.boxes:
                file.write(f"{box.name},,,,,,,,,\n")
                for m in box.materials:
                    file.write(f"{m.name},{m.depth},{m.description},{m.quantity},{m.datasheet},{m.supplier},{m.package},{m.x},{m.y},{m.z}\n")
                file.write(",,,,,,,,,\n")

# FIN