# -*- coding: utf-8 -*-
"""
LBB: Lesson Class

@author: kampff
"""

# Import modules
import LBB.Engine.utilities as Utilities
import LBB.Engine.instruction as Instruction
import LBB.Engine.image as Image
import LBB.Engine.video as Video
import LBB.Engine.task as Task

# Lesson Class
class Lesson:
    """
    LBB Lesson Class

    Stores a link to a video tutorial (optional) and a list of steps to complete the lesson
    """ 
    def __init__(self, text=None, dictionary=None):
        self.index = None               # Lesson index
        self.name = None                # Lesson name
        self.slug = None                # Lesson slug
        self.depth = None               # Lesson depth
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
            "depth": self.depth,
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
        self.depth = dictionary.get("depth")
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

        # Extract name and slug
        self.name = text[line_count].split(':')[1].strip()
        self.slug = self.name.lower().replace(' ', '-')
        line_count += 1

        # Extract description
        self.description = text[line_count]
        line_count += 1

        # List lesson depths
        depths = []
        if self.depth == "01":
            depths.append("01")
        elif self.depth == "10":
            depths.append("01")
            depths.append("10")
        elif self.depth == "11":
            depths.append("01")
            depths.append("10")
            depths.append("11")
        else:
            print(f"Invalid Lesson Depth Level: {self.depth}")
            exit(-1)

        # Extract video
        video_url = text[line_count].split('(')[1][:-1]
        if video_url != '':
            self.video = Video.Video(f"[{self.name}]({video_url})")
        line_count += 1

        # Find lesson section
        line_count = Utilities.find_line(text, "## Lesson")
        line_count += 1

        # Load steps
        self.steps = []
        step_count = 0
        while line_count < max_count:
            step_depth = Utilities.get_depth_from_symbol(text[line_count][0])
            step_text = text[line_count][2:]
            # Classify step
            if step_text.startswith("**TASK**"):
                task_text = []
                task_text.append(step_text)
                line_count += 1
                # Extract task steps
                while not text[line_count].startswith(">"):
                    task_text.append(text[line_count].strip())
                    line_count += 1
                task_text.append(text[line_count])
                task = Task.Task(task_text)
                task.index = step_count
                task.depth = step_depth
                if step_depth in depths:
                    self.steps.append(task)
            elif step_text.startswith('!['):
                image = Image.Image(step_text)
                image.index = step_count
                image.depth = step_depth
                if step_depth in depths:
                    self.steps.append(image)
            else:
                instruction = Instruction.Instruction(step_text)
                instruction.index = step_count
                instruction.depth = step_depth
                if step_depth in depths:
                    self.steps.append(instruction)
            step_count += 1
            line_count += 1
        return

    # Render lesson object as Markdown or HTML
    def render(self, type="MD"):
        output = []
        if type == "MD":
            if self.video:
                output.append(f"#### Watch this video: {self.video.render(type)}\n")
            else:
                output.append(f"### {self.name}\n")
            output.append(f"> {self.description}\n\n")
        elif type == "HTML":
            output.append(f"<h3>{self.name}</h3")
            output.append(f"{self.description}<br>")
            if self.video:
                output.extend(self.video.render(type))
        for step in self.steps:
            output.extend(step.render(type=type))
        if type == "MD":
            output.append("\n")
        elif type == "HTML":
            output.append("<hr>")
        return output

#FIN