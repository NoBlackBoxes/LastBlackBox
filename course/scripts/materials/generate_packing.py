# -*- coding: utf-8 -*-
"""
Generate packing lists from course BOMs

@author: kampff
"""

# Imports
import os
import numpy as np
import pandas as pd
import LBB.config as Config
import LBB.utilities as Utilities
import LBB.Engine.course as Course

# Specify Number of Kits
num_kits = 15

# List courses
course_names = ["The Last Black Box", "Bootcamp", "Build a Brain"]
for course_name in course_names:
    course = Course.Course(course_name)
    BOM_path = f"{Config.course_root}/_resources/materials/BOM/{course.slug}_BOM.csv"
    bom = pd.read_csv(BOM_path)

    # Insert additional fields
    bom.insert(4, '#kits', int(num_kits))
    bom.insert(5, '#required', num_kits*bom['quantity'].astype(int))
    bom.insert(6, '#available', 0)
    bom.insert(7, '#order', 0)
    bom.insert(8, '#ordered', 0)

    # Save to packing file
    packing_path = f"{Config.course_root}/_resources/materials/packing/{course.slug}_packing_list.csv"
    bom.to_csv(packing_path, index=False)  # Set index=False if you don't want to include the index in the CSV

    # Group packages
    grouped = bom.groupby('package')
    packages = {}
    for name, package in grouped:
        packages[name] = package

    # Estimate package volumes
    total_volume = 0.0
    print("\n")
    print(f"{course.name}")
    print(f"----------------")
    for name in packages:
        package = packages[name]
        lengths = package['x']
        widths = package['y']
        heights = package['z']
        volume = np.sum(lengths * widths * heights)
        print(f"{name}: {volume}")
        total_volume = total_volume + volume
    print(f"----------------")
    print(f"Total: \t\t{total_volume:.0f} mm^3")
    print(f"Available: \t{280 * 220 * 110:.0f} mm^3")

#FIN
