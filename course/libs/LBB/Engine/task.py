# -*- coding: utf-8 -*-
"""
LBB: Task Class

@author: kampff
"""

# Import modules
import LBB.Engine.utilities as Utilities
import LBB.Engine.instruction as Instruction
import LBB.Engine.image as Image
import LBB.Engine.code as Code

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
        self.steps = Utilities.extract_steps_from_dict(dictionary)
        self.target = dictionary.get("target")
        return
    
    # Parse task string
    def parse(self, text):
        # Set line counter
        line_count = 0

        # Extract description
        self.description = text[line_count].split(":")[1].strip()
        line_count += 1

        # Extract task steps
        self.steps = []
        step_count = 0
        while not text[line_count].startswith("> "):
            line_count, step = Utilities.extract_step_from_text(text, line_count)
            step.index = step_count
            self.steps.append(step)
            step_count += 1

        # Extract target
        self.target = text[line_count][2:]
        return

    # Render task object as Markdown or HTML
    def render(self, type="MD"):
        output = []
        if type == "MD":
            output.append(f"**TASK**: {self.description}\n")
        elif type == "HTML":
            html = Utilities.convert_emphasis_tags(self.description)
            html = Utilities.convert_markdown_links(html)
            output.append(f"{html}")
        for step in self.steps:
            for line in step.render(type=type):
                output.append(line)
        if type == "MD":
            output.append(f"<details><summary><strong>Target</strong></summary>\n")
            output.append(f":-:-: {self.target}\n")
            output.append(f"</details><hr>\n\n")
        elif type == "HTML":
            html = Utilities.convert_emphasis_tags(self.target)
            html = Utilities.convert_markdown_links(html)
            output.append(f"{html}\n")
        return output

#FIN