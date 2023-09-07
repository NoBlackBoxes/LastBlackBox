#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 - Insert goal list into each box's README.md
"""
import os

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
boxes_path = repo_path + '/boxes'

#
# Helper functions
#

# Find goals section in README file
def find_goals_section(readme):
    # Find start and end of goals section
    start = stop = lc = 0
    found = False
    for line in readme:
        if (line[:4] == '----'):
            if (found):
                stop = lc
                break
        elif((not found) and (line[:8] == '## Goals')):
            start = lc - 1
            found = True
        lc = lc + 1
    return start, stop

# Insert goal list into README file
def insert_goals(box):
    # Store paths
    box_path = boxes_path + "/" + box
    goals_path = box_path + "/goals.md"
    topics_path = box_path + "/topics.md"
    readme_path = box_path + "/README.md"
    print(readme_path)

    # Read README.md
    with open(readme_path, encoding='utf8') as f:
        readme = f.readlines()

    # Read goals.md
    with open(goals_path, encoding='utf8') as f:
        goals = f.readlines()

    # Read topics.md
    with open(topics_path, encoding='utf8') as f:
        topics = f.readlines()

    # Find start and end of goals section and split readme
    goals_section = ['## Goals\n'] + ['\n'] + goals + ['\n\n']
    start, stop = find_goals_section(readme)
    pre_readme = readme[:(start + 1)]
    post_readme = readme[stop:]

    # Insert goals
    readme = pre_readme + goals_section + post_readme

    # Store README.md
    f = open(readme_path, 'w', encoding='utf8')
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
    'reflexes',
    'power',
    'data',
    'logic',
    'memory',
    'fpgas',
    'computers',
    'control',
    'behaviour',
    'systems',
    'networks',
    'security',
    'audio',
    'vision',
    'learning',
    'intelligence',
    'python',
    'websites'
]

# Insert goals table into README
for box in boxes:
    insert_goals(box)

#FIN
