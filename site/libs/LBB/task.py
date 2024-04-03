# -*- coding: utf-8 -*-
"""
LBB: Task Class

@author: kampff
"""

# Import libraries
import os

# Import modules
import LBB.input as Input

# TO DO!
# - Add page footers (link to next topic...complete box...etc.)
# - Add progress header (and course indicator...)

# Task Class
class Task:
    def __init__(self, text=None):
        self.name = None            # name
        self.descriptions = None    # descriptions
        self.inputs = None          # inputs
        if text:
            self.parse(text)
        return
    
    def parse(self, text):
        semicolon_split = text.split(':')
        self.name = semicolon_split[0].split('(')[-1][:-1]
        right_bracket_split = semicolon_split[1][1:].split(']')
        self.descriptions = []
        self.inputs = []
        for chunk in right_bracket_split:
            left_bracket_split = chunk.split('[')
            if len(left_bracket_split) > 1:
                self.descriptions.append(left_bracket_split[0])
                self.inputs.append(Input.Input(left_bracket_split[1]))
            else:
                self.descriptions.append(left_bracket_split[0])
                self.inputs.append(Input.Input("none"))
        return
    
    def render(self):
        output = []
        output.append(f"<h4>TASK:</h4><i>{self.name}</i><br>")
        output.append(f"<form method=post enctype=multipart/form-data>")
        num_subtasks = len(self.descriptions)
        for i in range(num_subtasks):
            output.append(f"{self.descriptions[i]}")
            output.append(self.inputs[i].render())
        output.append("<br><input type=\"submit\" value=\"Submit\">")
        output.append(f"</form>")
        return "".join(output)

#FIN