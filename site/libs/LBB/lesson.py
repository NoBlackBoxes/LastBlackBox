# -*- coding: utf-8 -*-
"""
LBB: Lesson Class

@author: kampff
"""

# Import libraries
import os

# Import modules
import LBB.task as Task

# Topic Class
class Lesson:
    def __init__(self, text=None):
        self.level = None           # level
        self.instructions = None    # instructions
        self.images = None          # images
        self.videos = None          # videos
        self.tasks = None           # tasks
        if text:
            self.parse_text(text)
        return

    def parse_text(self, text):
        # Extract level
        if text[0] == f"{{01}}":
            self.level = 1
        elif text[0] == f"{{10}}":
            self.level = 2
        elif text[0] == f"{{11}}":
            self.level = 3
        else:
            print(f"Malformed level indicator during lesson parsing: {text[0]} vs {{01}}")
            return

        # Extract instructions
        self.instructions = []
        line_count = 1
        max_count = len(text)
        while line_count < max_count:
            if text[line_count][0] != '\n':
                self.instructions.append(text[line_count][:-1])
            line_count += 1
    
        # Extract images
        # Extract videos
        # Extract tasks

        return

#FIN