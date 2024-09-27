# -*- coding: utf-8 -*-
"""
LBB: Project Class

@author: kampff
"""

# Import libraries

# Import modules

# Project Class
class Project:
    def __init__(self, text=None):
        self.name = None            # name
        self.description = None     # description
        if text:
            self.parse(text)
        return

    def parse(self, text):
        # Line counter
        line_count = 0
        max_count = len(text)

         # Extract name
        self.name = text[0]

        # Extract description
        self.description = []
        line_count = 1
        while text[line_count][0] != '{':
            if text[line_count][0] != '\n':
                self.description.append(text[line_count])
            line_count += 1
        self.description = "".join(self.description)
        
    def render(self):
        output = ''
        return output
#FIN