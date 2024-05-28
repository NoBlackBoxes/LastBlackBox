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

# Topic Class
class Lesson:
    def __init__(self, text=None):
        self.level = None       # level
        self.steps = None       # steps
        if text:
            self.parse(text)
        return

    def parse(self, text):
        # Extract level
        if text[0] == f"{{01}}":
            self.level = 1
        elif text[0] == f"{{10}}":
            self.level = 2
        elif text[0] == f"{{11}}":
            self.level = 3
        else:
            print(f"Malformed level indicator during lesson parsing: {text[0]} vs {{01}}")
            return

        # Extract steps
        self.steps = []
        line_count = 1
        max_count = len(text)
        while line_count < max_count:
            if text[line_count][0] != '\n':
                # Classify step
                if text[line_count][0:4] == '- *V':
                    video = Video.Video(text[line_count])
                    self.steps.append(video)
                elif text[line_count][0:5] == '- **T':
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