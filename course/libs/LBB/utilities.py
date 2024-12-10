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
        depth = 1
    elif symbol == '+':
        depth = 2
    elif symbol == '*':
        depth = 3
    else:
        print(f"Unrecognized depth symbol ({symbol}) in lesson text.")
        exit(-1)
    return depth

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

#FIN