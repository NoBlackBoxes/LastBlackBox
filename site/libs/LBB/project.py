# -*- coding: utf-8 -*-
"""
LBB: Project Class

@author: kampff
"""

# Import libraries

# Import modules
import LBB.instruction as Instruction
import LBB.image as Image
import LBB.video as Video

# Project Class
class Project:
    def __init__(self, text=None):
        self.name = None            # project name
        self.steps = None           # project steps
        if text:
            self.parse(text)
        return

    def parse(self, text):
        # Line counter
        line_count = 1
        max_count = len(text)

         # Extract name
        self.name = text[1][4:-1]

        # Extract steps
        self.steps = []
        line_count = 1
        max_count = len(text)
        while line_count < max_count:
            if text[line_count][0] != '\n':
                # Classify step
                if text[line_count][0:8] == '- *Watch':
                    video = Video.Video(text[line_count])
                    self.steps.append(video)
                elif text[line_count][0:4] == '<img':
                    image = Image.Image(text[line_count])
                    self.steps.append(image)
                else:
                    instruction = Instruction.Instruction(text[line_count])
                    self.steps.append(instruction)
            line_count += 1
        return
        
    def render(self):
        output = ''
        return output
#FIN