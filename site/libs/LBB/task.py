# -*- coding: utf-8 -*-
"""
LBB: Task Class

@author: kampff
"""

# Import modules
import LBB.utilities as Utilities
import LBB.instruction as Instruction
import LBB.image as Image

# Task Class
class Task:
    def __init__(self, text=None, dictionary=None):
        self.index = None               # Step index
        self.type = "task"              # Step type
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
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "target": self.target
        }
        return dictionary

    # Convert dictionary to task object
    def from_dict(self, dictionary):
        self.index = dictionary.get("index")
        self.type = dictionary.get("type")
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
        self.description = text[0][16:]
        self.description = Utilities.convert_emphasis_tags(self.description)
        self.description = Utilities.convert_markdown_links(self.description)
        line_count += 1

        # Extract steps
        self.steps = []
        step_count = 0
        while not text[line_count].startswith("<details>"):
            # Classify step
            if text[line_count].startswith('<p align="center">'):
                image = Image.Image(text[line_count+1])
                image.index = step_count
                self.steps.append(image)
                line_count += 2
            else:
                instruction = Instruction.Instruction(text[line_count].strip())
                instruction.index = step_count
                self.steps.append(instruction)
            step_count += 1
            line_count += 1
        line_count += 1

        # Extract target
        self.target = text[line_count]
        self.target = Utilities.convert_emphasis_tags(self.target)
        self.target = Utilities.convert_markdown_links(self.target)
        return

#FIN