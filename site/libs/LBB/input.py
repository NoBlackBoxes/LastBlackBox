# -*- coding: utf-8 -*-
"""
LBB: Input Class

@author: kampff
"""

# Import libraries
import os

# Import modules

# Input Class
class Input:
    def __init__(self, _type, _name):
        self.type = _type        # type
        self.name = _name        # name
        self.value = None        # value
        
    def render(self):
        output = []
        if self.type == "number":
            output.append(f"\t<input type=\"text\" class=\"task_input_text\" name = \"{self.name}\" required>")
        if self.type == "photo":
            output.append(f"<input type=\"file\" class=\"task_input_photo\" name = \"{self.name}\" accept=\"image/x-png,image/jpeg,image/gif\" required>")
        return "".join(output)

#FIN