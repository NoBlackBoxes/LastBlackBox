# -*- coding: utf-8 -*-
"""
Generate a session README from a template

@author: kampff
"""

# Imports
import os
import glob
import LBB.utilities as Utilities
import LBB.config as Config
import LBB.Engine.session as Session

# Reload libraries
import importlib
importlib.reload(Utilities)
importlib.reload(Config)
importlib.reload(Session)

# Specify course/session
#course_session_basename = "/versions/buildabrain/01_sensing-the-world"
#course_session_basename = "/versions/buildabrain/02_making-things-move"
#course_session_basename = "/versions/buildabrain/03_digital-decisions"
#course_session_basename = "/versions/buildabrain/04_how-computers-work"
#course_session_basename = "/versions/buildabrain/05_build-a-robot"
#course_session_basename = "/versions/buildabrain/06_the-software-stack"
#course_session_basename = "/versions/buildabrain/07_how-the-internet-works"
course_session_basename = "/versions/buildabrain/08_artificial-intelligence"
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