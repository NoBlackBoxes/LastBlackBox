#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Extract box READMEs to Flask "templates" folder
"""
import os
import shutil

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
boxes_path = repo_path + '/boxes'
site_path = repo_path + '/site'
templates_path = site_path + '/templates'
template_boxes_path = templates_path + '/boxes'

# Create boxes folders (delete existing)
if not os.path.exists(template_boxes_path):
    os.makedirs(template_boxes_path)
else:
    shutil.rmtree(template_boxes_path)
    os.makedirs(template_boxes_path)

# Extract README files
def extract_README(box):
    # Store paths
    box_path = boxes_path + "/" + box
    readme_path = box_path + "/README.md"
    template_box_path = template_boxes_path + "/" + box
    template_readme_path = template_box_path + "/README.md"
    print(readme_path)

    # Read README.md
    with open(readme_path, encoding='utf8') as f:
        readme = f.readlines()

    # Process README here!

    # Create template box folder
    os.makedirs(template_box_path)

    # Store README.md in templates folder
    f = open(template_readme_path, 'w', encoding='utf8')
    f.writelines(readme)
    f.close()

    return

#
# Script
#

# List all "boxes" in order of processing (and placement in BOM)
boxes = [
    'atoms',
    'electrons',
    'magnets',
    'light',
    'sensors',
    'motors',
    'transistors',
    'amplifiers',
    'circuits',
    'power',
    'data',
    'logic',
    'memory',
    'fpgas',
    'computers',
    'control',
    'robotics',
    'systems',
    'linux',
    'python',
    'networks',
    'websites',
    'servers',
    'security',
    'audio',
    'vision',
    'learning',
    'intelligence'
]

# Extract (and pre-process) box READMEs for website rendering
for box in boxes:
    extract_README(box)

#FIN
