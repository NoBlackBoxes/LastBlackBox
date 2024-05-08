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
                input_descriptor = left_bracket_split[1].split('.')
                input_type = input_descriptor[0]
                if len(input_descriptor) > 1:
                    input_name = input_descriptor[1]
                    self.inputs.append(Input.Input(input_type, input_name))
                else:
                    self.inputs.append(Input.Input(input_type, input_type))
            else:
                self.descriptions.append(left_bracket_split[0])
                self.inputs.append(Input.Input("none", "none"))
        return
    
    def render(self):
        output = []
        output.append("<div id=\"task_box\">\n")
        output.append("<h4 id=\"task_label\">TASK</h4>\n")
        output.append(f"<h3 id=\"task_name\">{self.name}</h3><br>\n")
        output.append("<form id=\"task_form\" method=post enctype=multipart/form-data>")
        num_subtasks = len(self.descriptions)
        for i in range(num_subtasks):
            output.append(f"{self.descriptions[i]}")
            if self.inputs[i].type == "photo":
                output.append("<br><br>")
            output.append(self.inputs[i].render())
        output.append("<br><input id=\"task_submit\" type=\"submit\" value=\"Submit\">")
        output.append(f"</form>")
        output.append(f"</div>\n")
        return "".join(output)

#FIN