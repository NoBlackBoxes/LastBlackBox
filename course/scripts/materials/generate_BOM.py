# -*- coding: utf-8 -*-
"""
Generate LBB Bill of Materials (BOM)

@author: kampff
"""

# Import Libraries
import os
import glob
import pandas as pd

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

    # Gather all materials
    materials = []
    for session in course.sessions:
        for box in session.boxes:
            for m in box.materials:
                materials.append(m)
    
    # Build dataframe
    dataframe = pd.DataFrame([m.to_dict() for m in materials])

    # Remove empty rows and label
    filtered = dataframe[(dataframe['quantity'] != 0) & (dataframe['quantity'].notna())]

    # Combine (aggregate) duplicates
    aggregations = {
        'name': 'first',            # First value
        'depth': 'first',           # First value
        'description': 'first',     # First value
        'quantity': 'sum',          # Sum the quantities
        'package': 'first',         # First value
        'datasheet': 'first',       # First value
        'supplier': 'first',        # First value
        'x': 'first',               # First value
        'y': 'first',               # First value
        'z': 'first',               # First value
        'unit_price': 'first',      # First value
        'bulk_price': 'first',      # First value
    }
    combined = filtered.groupby('name', as_index=False).agg(aggregations)

    # Sort by packages
    sorted = combined.sort_values(['package', 'name'])

    # Generate BOM
    BOM_path = f"{Config.course_root}/_resources/materials/BOM/{course.slug}_BOM.csv"
    sorted.to_csv(BOM_path, index=False)  # Set index=False if you don't want to include the index in the CSV

# FIN