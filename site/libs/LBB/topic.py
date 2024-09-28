# -*- coding: utf-8 -*-
"""
LBB: Topic Class

@author: kampff
"""

# Import libraries

# Import modules
import LBB.instruction as Instruction
import LBB.image as Image
import LBB.video as Video
import LBB.task as Task

# Topic Class
class Topic:
    def __init__(self, text=None):
        self.name = None            # topic name
        self.steps = None           # topic steps
        if text:
            self.parse(text)
        return

    def parse(self, text):
        # Line counter
        line_count = 0
        max_count = len(text)

         # Extract name
        self.name = text[0][:-1]
        
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
                elif text[line_count][0:10] == '  - **Task':
                    task = Task.Task(text[line_count])
                    self.steps.append(task)
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
        for step in self.steps:
            output = output + step.render()
        return output
#FIN