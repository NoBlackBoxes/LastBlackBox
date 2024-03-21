# -*- coding: utf-8 -*-
"""
LBB: Task Class

@author: kampff
"""

# Import libraries
import os

# Import modules

# Task Class
class Task:
    def __init__(self, text=None):
        self.name = None            # box name
        self.description = None     # box description
        self.tasks = None     # box description
        if text:
            self.parse_text(text)
        return
    
    def parse_text(text):

        return

#FIN