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
        self.description = None         # Task description
        self.result= None               # Task result
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
        dictionary.update({"description": self.description})
        dictionary.update({"result": self.result})
        return dictionary

    # Convert dictionary to task object
    def from_dict(self, dictionary):
        self.index = dictionary.get("index")
        self.type = dictionary.get("type")
        self.description = dictionary.get("description")
        self.result = dictionary.get("result")
        return
    
    # Parse task string
    def parse(self, text):
        description_string = text[0]
        result_string = text[1]
        self.description = description_string[16:]
        self.result = result_string[2:]
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