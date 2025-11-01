# -*- coding: utf-8 -*-
"""
LBB : Config

@author: kampff
"""

# Imports
import os

# Get username
username = os.getlogin()

# Load LBB configuration variables
repo_root = f"/home/{username}/NoBlackBoxes/LastBlackBox"
boxes_root = repo_root + "/boxes"
course_root = repo_root + "/course"

# Store course version names
course_names = ["The Last Black Box", "Bootcamp", "Braitenberg", "Build a Brain", "AI-Workshops"]

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

# Store box names (and order) - Bootcamp
bootcamp_box_names = [
    'Atoms',
    'Electrons',
    'Magnets',
    'Sensors',
    'Motors',
    'Transistors',
    'Data',
    'Logic',
    'Memory',
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
    'Audio',
    'Vision'
]

# Store box names (and order) - Braitenberg
braitenberg_box_names = [
    'Atoms',
    'Electrons',
    'Magnets',
    'Sensors',
    'Motors',
    'Transistors',
    'Data',
    'Logic',
    'Memory',
    'Computers',
    'Control',
    'Robotics'
]

# Store box names (and order) - Build a Brain
buildabrain_box_names = [
    'Sensors',
    'Motors',
    'Transistors',
    'Computers',
    'Robotics',
    'Systems',
    'Networks',
    'Intelligence'
]

# Store box names (and order) - Own Phone
ownphone_box_names = [
    'Electrons',
    'Magnets',
    'Light',
    'Transistors',
    'Data',
    'Computers',
    'Systems',
    'Networks',
    'Security'
]

# Specify package box names (7 x 7 grid)

# Large (6 x 4)
large_box_names = [
    'Atoms',
    'Electrons',
    'Magnets',
    'Light',
    'Amplifiers',
    'Circuits',
    'Data',
    'Logic',
    'Control',
    'Robotics',
    'Power',
    'Systems',
    'Websites',
    'Servers',
    'Security',
    'Audio',
    'Atoms',
    'Electrons',
    'Magnets',
    'Light',
    'Amplifiers',
    'Circuits',
    'Data',
    'Logic'
]

# Medium (4 x 3)
medium_box_names = [
    'Sensors',
    'Motors',
    'Transistors',
    'Memory',
    'FPGAs',
    'Computers',
    'Linux',
    'Python',
    'Networks',
    'Vision',
    'Learning',
    'Intelligence'
]

# Small (2 x 3)
small_box_names = [
    'Sensors',
    'Motors',
    'Transistors',
    'Memory',
    'FPGAs',
    'Computers',
    'Linux',
    'Python',
    'Networks'
]

# Cables (1 x 7)
cables_box_names = [
    'Websites',
    'Servers',
    'Security',
    'Audio',
    'Vision',
    'Learning',
    'Intelligence',
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
    'Computers'
]

#FIN
