#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 - Generate a bill of materials from the content of all "boxes"
 - Insert contents table into each box's README.md
"""
import os

# Get LBB root
LBBROOT = os.environ.get('LBBROOT')

# Specify LBB boxes folder
LBBBOXES = LBBROOT + '/repo/boxes'

#
# Helper functions
#

# Find materials section in README file
def find_materials_section(readme):
    # Find start and end of materials section
    start = stop = lc = 0
    found = False
    for line in readme:
        if(line[:4] == '----'):
            if(found):
                stop = lc
                break
            else:        
                start = lc
                found = True
        lc = lc + 1
    return start, stop

# Convert materials.csv to markdown table
def convert_materials(materials):
    contents = []
    required = []
    is_contents = True
    for line in materials[1:]:
        # Remove new line
        line = line[:-1]

        # Seperate fields
        fields = line.split(',')

        # Skip blanks
        if(fields[0] == ''):
            continue

        # Split contents and required
        if(fields[0] == 'required'):
            is_contents = False
            continue

        # Edit Data and Link fields into markdown links
        if(is_contents): # contents
            if(fields[3] != ''):
                fields[3] = "[-D-]("+fields[3]+")"
            else:
                fields[3] = '-'
            if(fields[4] != ''):
                fields[4] = "[-L-]("+fields[4]+")"
            else:
                fields[4] = '-'
            contents.append('|'.join(fields) + '\n')
        else: # required
            if(fields[3] != ''):
                fields[3] = "["+fields[3]+"](/boxes/"+fields[3]+"/README.md)"
            else:
                fields[3] = '-'
            required.append('|'.join(fields) + '\n')

    return contents, required

# Insert materials table into README file
def insert_materials(box):
    # Store paths
    box_path = LBBBOXES + "/" + box
    materials_path = box_path + "/materials.csv"
    readme_path = box_path + "/README.md"

    # Read README.md
    with open(readme_path) as f:
        readme = f.readlines()
    
    # Read materials.csv
    with open(materials_path) as f:
        materials = f.readlines()
    
    # Append final newline (if not present)
    if(materials[-1][-1] != '\n'):
        materials[-1] = materials[-1] + '\n'
    
    # Create "materials" header and footer
    materials_header = ["\n", "<details><summary><b>Materials</b></summary><p>\n", "\n"]
    materials_footer = ["\n", "</p></details>\n", "\n"]

    # Create "contents" header
    contents_header = ["Contents|Description| # |Data|Link|\n", ":-------|:----------|:-:|:--:|:--:|\n"]

    # Create "required" header
    required_header = ["\nRequired|Description| # |Box|\n",":-------|:----------|:-:|:-:|\n"]

    # Convert materials.csv to markdown table
    contents, required = convert_materials(materials)

    # Find start and end of materials section and split readme
    start, stop = find_materials_section(readme)
    pre_readme = readme[:(start+1)]
    post_readme = readme[stop:]

    # Insert materials (and table formatting)
    materials_section = materials_header + contents_header + contents + required_header + required + materials_footer
    readme = pre_readme + materials_section + post_readme

    # Store README.md
    f = open(readme_path, 'w')
    f.writelines(readme)
    f.close()

    return

# Append materials contents to BOM
def append_materials(BOM, box):
    # Store paths
    box_path = LBBBOXES + "/" + box
    materials_path = box_path + "/materials.csv"
    
    # Read materials.csv (contents only)
    materials = []
    f = open(materials_path, 'r')
    line = f.readline() # Skip contents header
    while True:
        line = f.readline()
        if(line[:8] == 'required'):
            break
        # Add line to materials
        materials.append(line)
        
    # Create "box" header
    box_header = [box + ",,,,\n"]

    # Append header and materials contents to BOM
    BOM = BOM + box_header + materials

    return BOM

#
# Script
#

# List all "boxes" in order of processing (and placement in BOM)
boxes = [
    'white',
    'electrons',
    'magnets',
    'light',
    'sensors',
    'motors',
    'amplifiers',
    'reflexes',
    'decisions',
    'data',
    'logic',
    'memory',
    'computers',
    'control',
    'behaviour',
    'systems',
    'networks',
    'hearing',
    'speech',
    'vision',
    'learning',
    'intelligence']

# TESTING
#boxes = ['_template']

# Insert materials table into README
for box in boxes:
    insert_materials(box)

# Generare BOM.csv by appending individual materials.csv (contents)
BOM = ['Name,Description,QTY,Datasheet,Supplier\n', ',,,,\n']
for box in boxes:
    BOM = append_materials(BOM, box)

# Store BOM
bom_path = LBBROOT + "/repo/course/materials/BOM.csv"
f = open(bom_path, 'w')
f.writelines(BOM)
f.close()

#FIN