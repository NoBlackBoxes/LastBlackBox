# -*- coding: utf-8 -*-
"""
LBB : Config

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
image_prefix = ""

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

# Store package names
package_names = [
    'Active',
    'Audio',
    'Cables',
    'Hardware',
    'Hindbrain',
    'Loose',
    'Magnets',
    'Mounts',
    'Passive',
    'Power'
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

#FIN