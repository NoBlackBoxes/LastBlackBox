# -*- coding: utf-8 -*-
"""
LBB: Utility Library

@author: kampff
"""

# Import libraries
import os
import glob
import shutil
import re

# Import modules
import LBB.Engine.instruction as Instruction
import LBB.Engine.image as Image
import LBB.Engine.task as Task
import LBB.Engine.code as Code

# Confirm folder (create if it does not exist)
def confirm_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    return

# Clear a folder (or create)
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

# Get depth from symbol
def get_depth_from_symbol(symbol):
    if symbol == '-':
        depth = "01"
    elif symbol == '+':
        depth = "10"
    elif symbol == '*':
        depth = "11"
    else:
        print(f"Unrecognized depth symbol ({symbol}) in lesson text.")
        exit(-1)
    return depth

# Get depths list
def get_depths(depth):
    depths = []
    if depth == "01":
        depths.append("01")
    elif depth == "10":
        depths.extend(["01", "10"])
    elif depth == "11":
        depths.extend(["01", "10", "11"])
    else:
        print(f"Invalid Box Depth Level: {depth}")
    return depths

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

# Find and convert all markdown emphasis tags
def convert_emphasis_tags(text):
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    return text

# Find and convert all markdown links in the format [text](url)
def convert_markdown_links(text):
    return re.sub(r'\[(.*?)\]\((.*?)\)', replace_link, text)

# Replace matched links (and convert internal repo links)
def replace_link(match):
    link_text, link_url = match.groups()
    if not link_url.startswith("http"):
        link_url = re.sub(r'^(?:\.\./)+', '', link_url)
        link_url = f"https://github.com/NoBlackBoxes/LastBlackBox/tree/master/{link_url}"
    return f'<a id="link" href="{link_url}" target="_blank">{link_text}</a>'

# Extract step from template text
def extract_step_from_text(text, line_count):
    step = None
    step_depth = get_depth_from_symbol(text[line_count].strip()[0])
    step_text = text[line_count][2:].strip()
    if step_text.startswith("**TASK**"):    # Task
        task_text = []
        task_text.append(step_text)
        line_count += 1
        # Extract task steps
        while not text[line_count].startswith(">"):
            task_text.append(text[line_count])
            line_count += 1
        task_text.append(text[line_count])
        task = Task.Task(task_text)
        task.depth = step_depth
        step = task
    elif step_text.startswith("!["):        # Image
        image = Image.Image(step_text)
        image.depth = step_depth
        step = image
    elif step_text.startswith("*code*"):     # Code
        code_text = []
        line_count += 1
        code_text.append(text[line_count])
        line_count += 1
        while not text[line_count].startswith("```"):
            code_text.append(text[line_count])
            line_count += 1
        code_text.append(text[line_count])
        code = Code.Code(code_text)
        code.depth = step_depth
        step = code
    elif step_text.startswith("```"):       # Debug
        print(text[:line_count])
        print(text[line_count])
        print("ERROR")
        exit(-1)
    else:                                   # Instruction
        instruction = Instruction.Instruction(step_text)
        instruction.depth = step_depth
        step = instruction
    line_count += 1
    return line_count, step

# Extract steps from dictionary
def extract_steps_from_dict(dictionary):
    steps = []
    for step_dictionary in dictionary.get("steps"):
        if step_dictionary.get("type") == "instruction":
            step = Instruction.Instruction(dictionary=step_dictionary)
        elif step_dictionary.get("type") == "image":
            step = Image.Image(dictionary=step_dictionary)
        elif step_dictionary.get("type") == "task":
            step = Task.Task(dictionary=step_dictionary)
        elif step_dictionary.get("type") == "code":
            step = Code.Code(dictionary=step_dictionary)
        else:
            print(f"Unknown step type in dictionary")
            exit(-1)
        steps.append(step)
    return steps


# Extract name and slug
def extract_lesson_name_and_slug(line):
    sections = line.split(':')
    if len(sections) == 3:
        name = f"NB3 : {sections[2].strip()}"
        slug = f"NB3_{sections[2].strip().lower().replace(' ', '-')}"
    else:
        name = sections[1].strip()
        slug = name.lower().replace(' ', '-')
    return name, slug

#FIN