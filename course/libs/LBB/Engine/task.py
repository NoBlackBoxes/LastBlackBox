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
            # Classify task step
            if text[line_count].startswith('!['):
                image = Image.Image(text[line_count])
                image.index = step_count
                self.steps.append(image)
            else:
                instruction = Instruction.Instruction(text[line_count])
                instruction.index = step_count
                self.steps.append(instruction)
            step_count += 1
            line_count += 1

        # Extract target
        self.target = text[line_count][2:]
        self.target = Utilities.convert_emphasis_tags(self.target)
        self.target = Utilities.convert_markdown_links(self.target)
        return

#FIN