# -*- coding: utf-8 -*-
"""
LBB : Engine : Lesson Class

@author: kampff
"""

# Imports
import LBB.utilities as Utilities
import LBB.Engine.instruction as Instruction
import LBB.Engine.image as Image
import LBB.Engine.video as Video
import LBB.Engine.task as Task
import LBB.Engine.code as Code

# Lesson Class
class Lesson:
    """
    LBB Lesson Class

    Stores a link to a video tutorial (optional) and a list of steps to complete the lesson
    """ 
    def __init__(self, _box, text=None, dictionary=None):
        self.course = _box.course       # Lesson parent (course)
        self.session = _box.session     # Lesson parent (session)
        self.box = _box                 # Lesson parent (box)
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
        self.video = Video.Video(self.box, dictionary=dictionary.get("video"))
        self.steps = Utilities.extract_steps_from_dict(dictionary)
        return

    # Parse lesson string
    def parse(self, text):
        # Set line counter
        line_count = 0
        max_count = len(text)

        # Extract name and slug
        self.name, self.slug = Utilities.extract_lesson_name_and_slug(text[line_count])
        line_count += 1

        # Extract description
        self.description = text[line_count]
        line_count += 1

        # Extract video
        video_url = text[line_count].split('(')[1][:-1]
        if video_url != '':
            self.video = Video.Video(self.box, f"[{self.name}]({video_url})")
        line_count += 1

        # Find lesson section
        line_count = Utilities.find_line(text, "## Lesson")
        line_count += 1

        # Extract lesson steps
        self.steps = []
        step_count = 0
        while line_count < max_count:
            line_count, step = Utilities.extract_step_from_text(self.course, text, line_count)
            step.index = step_count
            self.steps.append(step)
            step_count += 1
        return

    # Render lesson object as Markdown or HTML
    def render(self, type="MD"):
        output = []
        if type == "MD":
            if self.video:
                output.append(f"#### Watch this video: [{self.name}]({self.video.url})\n")
                output.append(f"{self.video.render(type)}\n")
            else:
                output.append(f"### {self.name}\n")
            output.append(f"> {self.description}\n\n")
        elif type == "HTML":
            output.append(f"<h3>{self.name}</h3")
            output.append(f"{self.description}<br>")
            if self.video:
                for line in self.video.render(type):
                    output.append(line)
        for step in self.steps:
            for line in step.render(type=type):
                output.append(line)
        if type == "MD":
            output.append("\n")
        elif type == "HTML":
            output.append("<hr>")
        return output

#FIN