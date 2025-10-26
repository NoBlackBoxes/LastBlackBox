# -*- coding: utf-8 -*-
"""
Generate CAD files for each LBB material

@author: kampff
"""

# Imports
import os
import glob
import re
import cadquery as cq
import LBB.config as Config
import LBB.utilities as Utilities
import LBB.Engine.course as Course
import LBB.Engine.box as Box
import LBB.Design.package as Package

# Build CAD packaging for each course
for course_name in Config.course_names:
    course = Course.Course(course_name)

    # Generate CAD object
    CAD_course_folder = f"{Config.course_root}/_resources/packaging/CAD/items/{course.slug}"
    Utilities.clear_folder(CAD_course_folder)
    for package_name in Config.package_names:
        Utilities.confirm_folder(f"{CAD_course_folder}/{package_name}")
    count = 0
    for session in course.sessions:
        for box in session.boxes:
            for m in box.materials:
                for n in range(m.quantity):
                    output_path = f"{CAD_course_folder}/{m.package}/{m.slug}_{box.slug}_{n}.step"
                    cuboid = cq.Workplane("XY").box(m.x, m.y, m.z)
                    Package.save_STEP(output_path, m.slug, cuboid)
                    count += 1
    print(f"{count} CAD materials generated for {course_name}")

#FIN