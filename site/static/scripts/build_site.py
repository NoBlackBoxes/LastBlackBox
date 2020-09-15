#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build lastblackbox.training from repo
"""
import os
import subprocess
import markdown
from pathlib import Path

# Get LBB root
LBBROOT = os.environ.get('LBBROOT')

# Get LBB repo
LBBREPO = LBBROOT + '/repo'

# Set site root
site_path = LBBROOT + '/repo/site/lastblackbox.training'

# Create site root (if not present)
ret = subprocess.call(["mkdir", "-p", site_path])

# Load header
header_path = html_path + '/header.html'
file = open(header_path, "r")
header_text = file.read()
file.close()

# Load footer
footer_path = html_path + '/footer.html'
file = open(footer_path, "r")
footer_text = file.read()
file.close()

# Find all markdown files
md_paths = list(Path(LBBREPO).rglob("*.md"))
print("Number of MD files to convert: %d" % (len(md_paths)))

# List all "boxes" in order of processing (and placement in BOM)
boxes = [
    'white', 'electrons', 'magnets', 'light', 'sensors', 'motors',
    'amplifiers', 'reflexes', 'decisions', 'data', 'logic', 'memory',
    'computers', 'control', 'behaviour', 'systems', 'networks', 'hearing',
    'speech', 'vision', 'learning', 'intelligence'
]

# Convert all markdown files to html
for path in md_paths:
    # Extract *.md stem path
    input_path = str(path)
    root, stem = input_path.split("LastBlackBox/repo")

    # Set *.html path
    output_path = site_path + stem[:-3] + '.html'

    # Make folders (as required)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load input markdown
    file = open(input_path, "r")
    text = file.read()
    file.close()

    # Parse markdown
    parsed = markdown.markdown(text)

    # Save output html
    file = open(output_path, "w")
    num_written = file.write(header_text)
    num_written = file.write(parsed)
    num_written = file.write(footer_text)
    file.close()

#FIN
