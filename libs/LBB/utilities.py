# -*- coding: utf-8 -*-
"""
LBB : Utilities

@author: kampff
"""

# Imports
import os, shutil, re

# Confirm folder (create if it does not exist)
def confirm_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    return

# Clear folder (or create)
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
    return

# List all sub-folder names in folder
def list_subfolder_names(folder_path):
    subfolder_names = []
    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        if os.path.isdir(path):
            subfolder_names.append(name)
    return subfolder_names

# Read text file, strip whitespace (including newline), and remove empty lines
def read_clean_text(path):
    with open(path, encoding='utf8') as f:
        lines = f.readlines()
    text = []
    is_code = False
    for line in lines:
        # Check for formatted code blocks
        if line.startswith("```"):
            is_code = not is_code
        if is_code:
            text.append(line.rstrip())      # Remove trailing whitespace (including /n)
        else:
            if line.strip():                # Remove empty lines
                text.append(line.rstrip())  # Remove trailing whitespace (including /n)
    return text

# Find line
def find_line(text, pattern):
    count = 0
    for line in text:
        if line.startswith(pattern):
            break
        count += 1
    return count

# Find lesson tags
def find_lesson_tags(text):
    tags = []
    for line in text:
        tags += re.findall(r'\{[^}]+\}', line)
    return tags

# Extract markdown image link
def extract_markdown_image_link(line):
    pattern = r'!\[[^\]]*\]\([^\)]*\)'
    match = re.search(pattern, line)
    return match.group(0) if match else None

# Extract name and slug
def extract_lesson_name_and_slug(line):
    sections = line.split(':')
    if len(sections) == 3:
        name = f"NB3 : {sections[2].strip()}"
        slug = f"NB3_{sections[2].strip().lower().replace(' ', '-').replace('(', '').replace(')', '')}"
    else:
        name = sections[1].strip()
        slug = name.lower().replace(' ', '-').replace('(', '').replace(')', '').replace('\'', '')
    return name, slug

#FIN