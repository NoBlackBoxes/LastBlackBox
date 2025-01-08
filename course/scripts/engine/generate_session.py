# -*- coding: utf-8 -*-
"""
Generate a session README from a template

@author: kampff
"""

# Import Libraries
import os
import glob

# Import modules
import LBB.Engine.utilities as Utilities
import LBB.Engine.config as Config
import LBB.Engine.session as Session

# Reload libraries and modules
import importlib
importlib.reload(Utilities)
importlib.reload(Config)
importlib.reload(Session)

# Config
Config.image_prefix = "../../../.."

# Specify course/session
#course_session_basename = "/versions/buildabrain/01_sensing-the-world"
#course_session_basename = "/versions/buildabrain/01_sensing-the-world"
course_session_basename = "/versions/buildabrain/03_digital-decisions"
session_template_path = f"{Config.course_root}/{course_session_basename}/_resources/template.md"

# Load session template
session_text = Utilities.read_clean_text(session_template_path)

# Build session
session = Session.Session(session_text)

# Render session README
README_text = session.render("Build a Brain", type="MD")

# Save README
README_path = f"{Config.course_root}/{course_session_basename}/README.md"
with open(README_path, 'w', encoding='utf8') as f:
    f.writelines(README_text)    

# FIN