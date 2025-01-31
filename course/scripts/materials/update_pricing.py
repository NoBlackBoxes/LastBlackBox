# -*- coding: utf-8 -*-
"""
Update pricing from BOM in local material.csv files

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

# Load LBB BOM
course_BOM_path = f"{Config.course_root}/_resources/materials/BOM/course_BOM.csv"
bom = pd.read_csv(course_BOM_path)

# Load LBB course
course = Course.Course("The Last Black Box")

# Update each box's material.csv files
for session in course.sessions:
    materials = session.boxes[0].materials
    if len(materials) == 0:
        continue
    for m in materials:
        # Update price for this material
        BOM_row = bom[bom['name'] == m.name].index[0]
        m.unit_price = bom.at[BOM_row, 'unit_price']
        m.bulk_price = bom.at[BOM_row, 'bulk_price']
    # Write updated materials.csv
    materials_path = f"{Config.boxes_root}/{session.boxes[0].slug}/_resources/materials.csv"
    dataframe = pd.DataFrame([m.to_dict() for m in materials])
    exclude_columns = ['slug']
    dataframe.drop(columns=exclude_columns).to_csv(materials_path, index=False, header=False)

# FIN