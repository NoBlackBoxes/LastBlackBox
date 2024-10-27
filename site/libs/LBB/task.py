# -*- coding: utf-8 -*-
"""
LBB: Task Class

@author: kampff
"""

# Import modules
import LBB.input as Input

# Task Class
class Task:
    def __init__(self, text=None, dictionary=None):
        self.index = None               # Step index
        self.type = "task"              # Step type
        self.name = None                # Task name
        self.descriptions = None        # Task descriptions
        self.inputs = None              # Task inputs
        if text:
            self.parse(text)            # Parse task from README text
        elif dictionary:
            self.from_dict(dictionary)  # Load task from dictionary
        return
    
    # Convert task object to dictionary
    def to_dict(self):
        dictionary = {}
        dictionary.update({"index": self.index})
        dictionary.update({"type": self.type})
        dictionary.update({"name": "TASK"})
        #dictionary.update({"descriptions": self.descriptions})
        return dictionary

    # Convert dictionary to task object
    def from_dict(self, dictionary):
        self.index = dictionary.get("index")
        self.type = dictionary.get("type")
        self.html = dictionary.get("name")
        return
    
    # Parse task string
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
        self.header(output)
        self.body(output)
        self.footer(output)
        return "".join(output)
    
    def header(self, output):
        output.append(f"{{% if task_status['{self.name}'] == 1 %}}\n")
        output.append(f"<div id=\"task_box_complete\">\n")
        output.append("{% else %}\n")
        output.append(f"<div id=\"task_box\">\n")
        output.append("{% endif %}\n")
        output.append(f"\t<h4 id=\"task_label\">TASK</h4>\n")
        output.append(f"\t<h3 id=\"task_name\">{self.name}</h3><br>\n")
        output.append(f"\t<form id=\"task_form\" method=post enctype=multipart/form-data>")
        return output

    def body(self, output):
        num_subtasks = len(self.descriptions)
        for i in range(num_subtasks):
            output.append(f"{self.descriptions[i]}")
            if self.inputs[i].type == "photo":
                output.append("<br><br>")
            output.append(self.inputs[i].render())
        return output

    def footer(self, output):
        output.append(f"\t<br><input id=\"task_submit\" type=\"submit\" value=\"Submit\">\n")
        output.append(f"\t\t<input type=\"hidden\" name=\"task_name\" value=\"{self.name}\"/>\n")        
        output.append(f"\t</form>\n")
        output.append(f"</div>\n")
        return output

#FIN