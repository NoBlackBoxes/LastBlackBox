# -*- coding: utf-8 -*-
"""
LBB: Task Class

@author: kampff
"""

# Import libraries
import os

# Import modules

# TO DO!
# - Parse task submission types (inline) and render task description accordingly.
# - Add page footers (link to next topic...complete box...etc.)
# - Add progress header (and course indicator...)

# Task Class
class Task:
    def __init__(self, text=None):
        self.name = None            # name
        self.description = None     # description
        self.type = None            # type
        if text:
            self.parse(text)
        return
    
    def parse(self, text):
        semicolon_split = text.split(':')
        task_description = semicolon_split[1][1:]
        self.name = task_description.split('(')[-1]
        self.description = task_description.split('(')[0]
        self.type = semicolon_split[-1][:-2]
        return
    
    def render(self):
        output = []
        output.append(f"<h3>TASK: {self.description}</h3>\n")
        return "".join(output)

#FIN