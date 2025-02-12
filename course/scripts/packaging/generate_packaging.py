# -*- coding: utf-8 -*-
"""
Generate LBB packaging descriptions

@author: kampff
"""

# Import libraries

# Import modules
import LBB.Engine.utilities as Engine_Utilities
import LBB.Engine.config as Config
import LBB.Packaging.utilities as Packaging_Utilities
import LBB.Packaging.package as Package

# Specify parameters
unit = 30           # Unit size (mm)
thick_wall = 1.5    # Fluted Thickness (mm)
thin_wall = 0.38    # Board Thickness (mm)

# Specify packages
packages = []
packages.append(Package.Package("Mailer", "FEFCO 0427",        "internal", 7*unit, 7*unit, 3.5*unit, "E-Flute", thick_wall, 0.0, "nearest"))
packages.append(Package.Package("Body",   "ECMA A20.20.03.01", "external", 7*unit, 0.5*unit, 7*unit, "GSM 300", thin_wall,  0.5, "down"))
packages.append(Package.Package("Large",  "ECMA A55.20.01.01", "external", 6*unit, 4*unit, 3*unit,   "GSM 300", thin_wall,  0.5, "down"))
packages.append(Package.Package("Medium", "ECMA A55.20.01.01", "external", 4*unit, 3*unit, 3*unit,   "GSM 300", thin_wall,  0.5, "down"))
packages.append(Package.Package("Small",  "ECMA A55.20.01.01", "external", 3*unit, 2*unit, 3*unit,   "GSM 300", thin_wall,  0.5, "down"))
packages.append(Package.Package("Cables", "ECMA A20.20.03.01", "external", 3*unit, 1*unit, 7*unit,   "GSM 300", thin_wall,  0.5, "down"))

# Save package descriptions
description_path = f"{Config.course_root}/_resources/packaging/package_dimensions.txt"
with open(description_path, "w") as file:
    for package in packages:
        dim = package.print_dimensions()
        file.write(dim)

# Save package models
model_folder = f"{Config.course_root}/_resources/packaging/CAD/boxes"
Engine_Utilities.clear_folder(model_folder)
for package in packages:
    model_path = f"{model_folder}/{package.name}.step"
    model = package.generate_model()
    Packaging_Utilities.save_STEP(model_path, package.name, model)

# Save package designs (printing)
design_folder = f"{Config.course_root}/_resources/packaging/designs"
Engine_Utilities.clear_folder(design_folder)
for package in packages:
    if package.name in ["Large", "Medium", "Small", "Cables"]:
        design_path = f"{design_folder}/{package.name}.svg"
        design = package.generate_design(unit, design_path)

# FIN