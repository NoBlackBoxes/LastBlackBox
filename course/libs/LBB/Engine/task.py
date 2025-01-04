# -*- coding: utf-8 -*-
"""
LBB: Task Class

@author: kampff
"""

# Import modules
import LBB.Engine.utilities as Utilities
import LBB.Engine.instruction as Instruction
import LBB.Engine.image as Image

# Task Class
class Task:
    def __init__(self, text=None, dictionary=None):
        self.index = None               # Step index
        self.type = "task"              # Step type
        self.depth = None               # Step depth
        self.description = None         # Task description
        self.steps = None               # Task steps
        self.target = None              # Task target
        if text:
            self.parse(text)            # Parse task from README text
        elif dictionary:
            self.from_dict(dictionary)  # Load task from dictionary
        return
    
    # Convert task object to dictionary
    def to_dict(self):
        dictionary = {
            "index": self.index,
            "type": self.type,
            "depth": self.depth,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "target": self.target
        }
        return dictionary

    # Convert dictionary to task object
    def from_dict(self, dictionary):
        self.index = dictionary.get("index")
        self.type = dictionary.get("type")
        self.depth = dictionary.get("depth")
        self.description = dictionary.get("description")
        self.steps = []
        for step_dictionary in dictionary.get("steps"):
            if step_dictionary.get("type") == "instruction":
                step = Instruction.Instruction(dictionary=step_dictionary)
            elif step_dictionary.get("type") == "image":
                step = Image.Image(dictionary=step_dictionary)
            else:
                print(f"Unknown step type in task: {self.description}")
                exit(-1)
            self.steps.append(step)
        self.target = dictionary.get("target")
        return
    
    # Parse task string
    def parse(self, text):
        # Set line counter
        line_count = 0
        
        # Extract description
        self.description = text[line_count][10:]
        self.description = Utilities.convert_emphasis_tags(self.description)
        self.description = Utilities.convert_markdown_links(self.description)
        line_count += 1

        # Extract task steps
        self.steps = []
        step_count = 0
        while not text[line_count].startswith("> "):
            step_depth = Utilities.get_depth_from_symbol(text[line_count][0])
            step_text = text[line_count][2:]
            # Classify task step
            if step_text.startswith('!['):
                image = Image.Image(step_text)
                image.index = step_count
                image.depth = step_depth
                self.steps.append(image)
            else:
                instruction = Instruction.Instruction(step_text)
                instruction.index = step_count
                instruction.depth = step_depth
                self.steps.append(instruction)
            step_count += 1
            line_count += 1

        # Extract target
        self.target = text[line_count][2:]
        return

    # Render task object as Markdown or HTML
    def render(self, type="MD"):
        output = []
        if type == "MD":
            output.append(f"**TASK**: {self.description}\n")
        elif type == "HTML":
            output.append(f"<task>")
        for step in self.steps:
            output.extend(f"{step.render(type=type)[0]}")
        if type == "MD":
            output.append(f"<details><summary><strong>Target</strong></summary>\n")
            output.append(f"{self.target}\n")
            output.append(f"</details><hr>\n\n")
        elif type == "HTML":
            html = Utilities.convert_emphasis_tags(self.target)
            html = Utilities.convert_markdown_links(html)
            output.append(f"{html}\n")
        return output

#FIN