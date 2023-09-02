#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 - Generate a bill of materials from the content (materials.csv) of all "boxes"
 - Insert contents table into each box's README.md
"""
import os

# Get LBB root -- KeyError if not found
try:
    LBBREPO = os.environ['LBBROOT']

except KeyError:
    LBBREPO = '../..'

# Specify LBB boxes folder
LBBBOXES = LBBREPO + '/boxes'

#
# Helper functions
#

# Find materials section in README file
def find_materials_section(readme):
    # Find start and end of materials section
    start = stop = lc = 0
    found = False
    for line in readme:
        if (line[:4] == '----'):
            if (found):
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
    for line in materials:
        # Remove new line
        line = line[:-1]

        # Seperate fields
        fields = line.split(',')

        # Skip blanks
        if (fields[0] == ''):
            continue

        # Edit Data and Link fields into markdown links
        if (fields[3] != ''):
            fields[3] = "[-D-](" + fields[3] + ")"
        else:
            fields[3] = '-'
        if (fields[4] != ''):
            fields[4] = "[-L-](" + fields[4] + ")"
        else:
            fields[4] = '-'
        contents.append('|'.join(fields) + '\n')

    return contents


# Insert materials table into README file
def insert_materials(box):
    # Store paths
    box_path = LBBBOXES + "/" + box
    materials_path = box_path + "/materials.csv"
    readme_path = box_path + "/README.md"
    print(readme_path)

    # Read README.md
    with open(readme_path, encoding='utf8') as f:
        readme = f.readlines()

    # Read materials.csv
    with open(materials_path, encoding='utf8') as f:
        materials = f.readlines()

    # Append final newline (if not present)
    if (materials[-1][-1] != '\n'):
        materials[-1] = materials[-1] + '\n'

    # Create "materials" header and footer
    materials_header = [
        "\n", "<details><summary><b>Materials</b></summary><p>\n", "\n"
    ]
    materials_footer = ["\n", "</p></details>\n", "\n"]

    # Create "contents" header
    contents_header = [
        "Contents|Description| # |Data|Link|\n",
        ":-------|:----------|:-:|:--:|:--:|\n"
    ]

    # Convert materials.csv to markdown table
    contents = convert_materials(materials)

    # Find start and end of materials section and split readme
    start, stop = find_materials_section(readme)
    pre_readme = readme[:(start + 1)]
    post_readme = readme[stop:]

    # Insert materials (and table formatting)
    materials_section = materials_header + contents_header + contents + materials_footer
    readme = pre_readme + materials_section + post_readme

    # Store README.md
    f = open(readme_path, 'w', encoding='utf8')
    f.writelines(readme)
    f.close()

    return


# Append materials contents to BOM
def append_materials(BOM, box):
    # Store paths
    box_path = LBBBOXES + "/" + box
    materials_path = box_path + "/materials.csv"

    # Read materials.csv
    materials = []
    f = open(materials_path, 'r', encoding='utf8')
    while True:
        line = f.readline()
        if (len(line) <= 2):
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
    'websites'
]

# Insert materials table into README
for box in boxes:
    insert_materials(box)

# Generate BOM.csv by appending individual materials.csv
BOM = ['Name,Description,QTY,Datasheet,Supplier\n', ',,,,\n']
for box in boxes:
    BOM = append_materials(BOM, box)

# Store BOM
bom_path = LBBREPO + "/course/materials/BOM.csv"
f = open(bom_path, 'w', encoding='utf8')
f.writelines(BOM)
f.close()

#FIN
