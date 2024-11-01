# -*- coding: utf-8 -*-
"""
LBB: Lesson Class

@author: kampff
"""

# Import modules
import LBB.utilities as Utilities
import LBB.instruction as Instruction
import LBB.image as Image
import LBB.video as Video
import LBB.task as Task

# Lesson Class
class Lesson:
    def __init__(self, text=None, dictionary=None):
        self.index = None               # Lesson index
        self.name = None                # Lesson name
        self.slug = None                # Lesson slug
        self.description = None         # Lesson description
        self.video = None               # Lesson video
        self.steps = None               # Lesson steps
        if text:
            self.parse(text)            # Parse lesson from README text
        elif dictionary:
            self.from_dict(dictionary)  # Load lesson from dictionary
        return

    # Convert lesson object to dictionary
    def to_dict(self):
        dictionary = {
            "index": self.index,
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "video": self.video.to_dict(),
            "steps": [step.to_dict() for step in self.steps]
        }
        return dictionary

    # Convert dictionary to lesson object
    def from_dict(self, dictionary):
        self.index = dictionary.get("index")
        self.name = dictionary.get("name")
        self.slug = dictionary.get("slug")
        self.description = dictionary.get("description")
        self.video = Video.Video(dictionary=dictionary.get("video"))
        self.steps = []
        for step_dictionary in dictionary.get("steps"):
            if step_dictionary.get("type") == "instruction":
                step = Instruction.Instruction(dictionary=step_dictionary)
            elif step_dictionary.get("type") == "image":
                step = Image.Image(dictionary=step_dictionary)
            elif step_dictionary.get("type") == "task":
                step = Task.Task(dictionary=step_dictionary)
            else:
                print(f"Unknown step type in lesson: {self.name}")
                exit(-1)
            self.steps.append(step)
        return

    # Parse lesson string
    def parse(self, text):
        # Set line counter
        line_count = 0
        max_count = len(text)

         # Extract name
        self.name = text[line_count].split('[')[1].split(']')[0]
        self.slug = self.name.lower().replace(' ', '-')

        # Extract video
        self.video = Video.Video(text[line_count])
        line_count += 1

        # Extract description
        self.description = Utilities.convert_emphasis_tags(text[line_count])
        line_count += 1

        # Extract steps
        self.steps = []
        step_count = 0
        while line_count < max_count:
            if text[line_count][0] != '\n':
                # Classify step
                if text[line_count].startswith('- [ ] **TASK'):
                    task_text = []
                    task_text.append(text[line_count])
                    line_count += 1
                    # Extract task steps
                    while not text[line_count].startswith("</details>"):
                        task_text.append(text[line_count])
                        line_count += 1
                    task_text.append(text[line_count])
                    task = Task.Task(task_text)
                    task.index = step_count
                    self.steps.append(task)
                elif text[line_count].startswith('<p align="center">'):
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
        return

#FIN