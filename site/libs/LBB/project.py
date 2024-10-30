# -*- coding: utf-8 -*-
"""
LBB: Project Class

@author: kampff
"""

# Import libraries

# Import modules
import LBB.instruction as Instruction
import LBB.image as Image

# Project Class
class Project:
    def __init__(self, text=None, dictionary=None):
        self.name = None                # Project name
        self.steps = None               # Project steps
        if text:
            self.parse(text)            # Parse project from README text
        elif dictionary:
            self.from_dict(dictionary)  # Load project from dictionary
        return

    # Convert project object to dictionary
    def to_dict(self):
        dictionary = {
            "name": self.name,
            "steps": [step.to_dict() for step in self.steps]
        }
        return dictionary

    # Convert dictionary to project object
    def from_dict(self, dictionary):
        self.name = dictionary.get("name")
        self.steps = []
        for step_dictionary in dictionary.get("steps"):
            if step_dictionary.get("type") == "instruction":
                step = Instruction.Instruction(dictionary=step_dictionary)
            elif step_dictionary.get("type") == "image":
                step = Image.Image(dictionary=step_dictionary)
            else:
                print(f"Unknown step type in project: {self.name}")
                exit(-1)
            self.steps.append(step)
        return

    # Parse project string
    def parse(self, text):
        # Set Line counter
        line_count = 0
        max_count = len(text)

         # Extract name
        line_count += 1
        self.name = text[1][4:-1]
        line_count += 1

        # Extract steps
        self.steps = []
        step_count = 0
        while line_count < max_count:
            if text[line_count][0] != '\n':
                # Classify step
                if text[line_count].startswith('<p align="center"><img'):
                    image = Image.Image(text[line_count])
                    image.index = step_count
                    self.steps.append(image)
                else:
                    instruction = Instruction.Instruction(text[line_count].strip())
                    instruction.index = step_count
                    self.steps.append(instruction)
                step_count += 1
            line_count += 1
        return
        
#FIN