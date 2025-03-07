# -*- coding: utf-8 -*-
"""
LBB : Engine : Project Class

@author: kampff
"""

# Imports
import LBB.utilities as Utilities
import LBB.Engine.instruction as Instruction
import LBB.Engine.image as Image
import LBB.Engine.video as Video
import LBB.Engine.task as Task
import LBB.Engine.code as Code


# Project Class
class Project:
    """
    LBB Project Class

    Steps to complete a session project
    """
    def __init__(self, depth, text=None, dictionary=None):
        self.index = None               # Project index
        self.name = None                # Project name
        self.slug = None                # Project slug
        self.depth = depth              # Project depth
        self.description = None         # Project description
        self.video = None               # Project video
        self.steps = None               # Project steps
        if text:
            self.parse(text)            # Parse project from README text
        elif dictionary:
            self.from_dict(dictionary)  # Load project from dictionary
        return

    # Convert project object to dictionary
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

    # Convert dictionary to project object
    def from_dict(self, dictionary):
        self.index = dictionary.get("index")
        self.name = dictionary.get("name")
        self.slug = dictionary.get("slug")
        self.depth = dictionary.get("depth")
        self.description = dictionary.get("description")
        self.video = Video.Video(dictionary=dictionary.get("video"))
        self.steps = Utilities.extract_steps_from_dict(dictionary)
        return

    # Parse project string
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

        # List project depths
        depths = Utilities.get_depths(self.depth)

        # Extract video
        video_url = text[line_count].split('(')[1][:-1]
        if video_url != '':
            self.video = Video.Video(f"[{self.name}]({video_url})")
        line_count += 1

        # Find lesson section
        line_count = Utilities.find_line(text, "## Lesson")
        line_count += 1

        # Extract project steps
        self.steps = []
        step_count = 0
        while line_count < max_count:
            line_count, step = Utilities.extract_step_from_text(text, line_count)
            if step.depth in depths:
                step.index = step_count
                self.steps.append(step)
                step_count += 1
        return

    # Render project object as Markdown or HTML
    def render(self, type="MD"):
        output = []
        if type == "MD":
            output.append(f"### {self.name}\n")
            output.append(f"> {self.description}\n\n")
        elif type == "HTML":
            output.append(f"<h3>{self.name}</h3")
            output.append(f"> {self.description}<br>")
        if self.video:
            if type == "MD":
                output.append(f"<details><summary><weak>Guide</weak></summary>\n")
                guide_string = f":-:-: A video guide to completing this project can be viewed <a href=\"{self.video.url}\" target=\"_blank\" rel=\"noopener noreferrer\">here</a>."
                output.append(f"{guide_string}\n")
                output.append(f"</details><hr>\n\n")
            elif type == "HTML":
                output.append(f"<HTML GUIDE>\n")
        for step in self.steps:
            output.extend(step.render(type=type))
        if type == "MD":
            output.append("\n")
        elif type == "HTML":
            output.append("<hr>")
        return output

#FIN