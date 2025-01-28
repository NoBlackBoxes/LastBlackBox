# -*- coding: utf-8 -*-
"""
Generate LBB packaging descriptions

@author: kampff
"""

# Import libraries

# Import modules
import LBB.Engine.config as Config
import LBB.Packaging.package as Package

# Specify parameters
unit = 30           # Unit size (mm)
thick_wall = 2      # Fluted Thickness (mm)
thin_wall = 0.38    # Board Thickness (mm)

# Specify packages
packages = []
packages.append(Package.Package("Mailer", "FEFCO 0427",        "internal", 7*unit, 7*unit, 3.5*unit, "E-Flute", thick_wall, 0.0, "nearest"))
packages.append(Package.Package("Body",   "ECMA A20.20.03.01", "external", 7*unit, 0.5*unit, 7*unit, "GSM 300", thin_wall,  1.5, "down"))
packages.append(Package.Package("Large",  "ECMA A55.20.01.01", "external", 6*unit, 4*unit, 3*unit,   "GSM 300", thin_wall,  1.5, "down"))
packages.append(Package.Package("Medium", "ECMA A55.20.01.01", "external", 4*unit, 3*unit, 3*unit,   "GSM 300", thin_wall,  1.5, "down"))
packages.append(Package.Package("Small",  "ECMA A55.20.01.01", "external", 3*unit, 2*unit, 3*unit,   "GSM 300", thin_wall,  1.5, "down"))
packages.append(Package.Package("Cables", "ECMA A20.20.03.01", "external", 3*unit, 1*unit, 7*unit,   "GSM 300", thin_wall,  1.5, "down"))

# Save package descriptions
description_path = f"{Config.course_root}/_resources/packaging/package_dimensions.txt"
with open(description_path, "w") as file:
    for package in packages:
        dim = package.print_dimensions()
        file.write(dim)

# FIN