# -*- coding: utf-8 -*-
"""
LBB: Utility Library

@author: kampff
"""

# Import libraries
import os
import shutil

# Render Header
def render_header(box, topic_index):
    header = []
    header.append(f"<!DOCTYPE html>\n<html>\n<body>\n")
    header.append(f"<title>LBB : {box.name} : {box.topics[topic_index].name}</title>\n")
    header.append(f"<h1>LBB : {box.name} : {box.topics[topic_index].name}</h1>\n")
    header.append(f"<h4>{box.topics[topic_index].description}</h4>\n")
    header.append(f"<hr>\n")
    return "".join(header)

# Render Footer
def render_footer(box, topic_index):
    footer = "\n</body>\n</html>"
    return footer

# Create a folder (if it does not exist)
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return

# Clear a folder (or create)
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
    return

#FIN