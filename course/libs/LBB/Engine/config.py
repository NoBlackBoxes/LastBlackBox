# -*- coding: utf-8 -*-
"""
LBB: Config Class

@author: kampff
"""

# Import libraries
import os

# Get username
username = os.getlogin()

# Load LBB configuration variables
repo_root = f"/home/{username}/NoBlackBoxes/LastBlackBox"
boxes_root = repo_root + "/boxes"
course_root = repo_root + "/course"

# Store box names (and order)
box_names = [
    'Atoms',
    'Electrons',
    'Magnets',
    'Light',
    'Sensors',
    'Motors',
    'Transistors',
    'Amplifiers',
    'Circuits',
    'Data',
    'Logic',
    'Memory',
    'FPGAs',
    'Computers',
    'Control',
    'Robotics',
    'Power',
    'Systems',
    'Linux',
    'Python',
    'Networks',
    'Websites',
    'Servers',
    'Security',
    'Audio',
    'Vision',
    'Learning',
    'Intelligence'
]

#FIN