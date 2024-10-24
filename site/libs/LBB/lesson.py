# -*- coding: utf-8 -*-
"""
LBB: Lesson Class

@author: kampff
"""

# Import libraries

# Import modules
import LBB.instruction as Instruction
import LBB.image as Image
import LBB.video as Video
import LBB.task as Task

# Lesson Class
class Lesson:
    def __init__(self, text=None):
        self.name = None            # lesson name
        self.description = None     # lesson description
        self.video = None           # lesson video
        self.steps = None           # lesson steps
        if text:
            self.parse(text)
        return

    def parse(self, text):
        # Set line counter
        line_count = 0
        max_count = len(text)

         # Extract name
        self.name = text[line_count].split('[')[1].split(']')[0]
        self.video = Video.Video(text[line_count])
        line_count += 1

        # Extract description
        self.description = text[line_count][:-1]
        line_count += 1

        # Extract steps
        self.steps = []
        max_count = len(text)
        while line_count < max_count:
            if text[line_count][0] != '\n':
                # Classify step
                if text[line_count][0:8] == '- **Task':
                    task = Task.Task(text[line_count])
                    self.steps.append(task)
                elif text[line_count][0:22] == '<p align="center"><img':
                    image = Image.Image(text[line_count])
                    self.steps.append(image)
                else:
                    instruction = Instruction.Instruction(text[line_count].strip())
                    self.steps.append(instruction)
            line_count += 1
        return

    def render(self):
        output = []
        output.append(self.video.render())
        for step in self.steps:
            output.append(step.render())
        output = "".join(output)
        return output

#FIN