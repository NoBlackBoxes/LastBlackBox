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
    header.append("<!DOCTYPE html>\n")
    header.append("<head>\n")
    header.append("{% include 'pwa.html' %}\n")
    header.append("<link rel=\"stylesheet\" type=\"text/css\" href=\"{{url_for('static', filename='styles/topic_style.css')}}\"/>\n")
    header.append("</head>\n\n")
    header.append("<html>\n<body>\n\n")
    header.append(f"<title>LBB : {box.name} : {box.topics[topic_index].name}</title>\n")
    header.append(f"<h2 id=\"box_name\">LBB : {box.name}</h2>\n")
    header.append(f"<h3 id=\"topic_name\">{box.topics[topic_index].name}</h3>\n")
    header.append(f"<h4 id=\"topic_description\">{box.topics[topic_index].description}</h4>\n")
    header.append(f"<hr>\n")
    return "".join(header)

# Render Footer
def render_footer(box, topic_index):
    footer = []
    footer.append("<hr>\n")
    if topic_index > 0:
        previous_topic_name = box.topics[topic_index-1].name.replace(" ", "_").lower()
        previous_topic = f"{previous_topic_name}"
        footer.append(f"<a href=\"{previous_topic}\">Previous Topic</a><br>\n")
    if topic_index < (len(box.topics)-1):
        next_topic_name = box.topics[topic_index+1].name.replace(" ", "_").lower()
        next_topic = f"{next_topic_name}"
        footer.append(f"<a href=\"{next_topic}\">Next Topic</a><br>\n")
    footer.append("</body>\n</html>")
    return "".join(footer)

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