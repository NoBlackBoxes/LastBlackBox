# -*- coding: utf-8 -*-
"""
LBB: Session Class

@author: kampff
"""

# Import libraries
import json

# Import modules
import LBB.Engine.utilities as Utilities
import LBB.Engine.box as Box
import LBB.Engine.project as Project

# Session Class
class Session:
    """
    LBB Session Class

    Stores a list of boxes opened during this session and the session project
    """
    def __init__(self, text=None, dictionary=None):
        self.index = None               # Session index
        self.name = None                # Session name
        self.slug = None                # Session slug (URL)
        self.description = None         # Session description
        self.boxes = None               # Session boxes
        self.project = None             # Session project
        if text:
            self.parse(text)            # Parse session from template text
        elif dictionary:
            self.from_dict(dictionary)  # Load session from dictionary
        return
    
    # Convert session object to dictionary
    def to_dict(self):
        dictionary = {
            "index": self.index,
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "boxes": [box.to_dict() for box in self.boxes],
            "project": self.project.to_dict()
        }
        return dictionary

    # Convert dictionary to session object
    def from_dict(self, dictionary):
        self.index = dictionary.get("index")
        self.name = dictionary.get("name")
        self.slug = dictionary.get("slug")
        self.description = dictionary.get("description")
        self.boxes = [Box.Box(dictionary=box_dictionary) for box_dictionary in dictionary.get("boxes", [])]
        self.project = Project.Project(dictionary=dictionary.get("project"))
        return

    # Parse session template
    def parse(self, text):
        # Set line counter
        line_count = 0
        max_count = len(text)

        # Extract name and slug
        title = text[line_count].strip()
        self.name = title.split(':')[1].strip()
        self.slug = self.name.lower().replace(' ', '-')
        line_count += 1

        # Extract description
        self.description = text[line_count]
        line_count += 1

        # Find boxes section
        line_count = Utilities.find_line(text, "## ")
        boxes_max_count = Utilities.find_line(text, "# Project") - 1

        # Load boxes
        self.boxes = []
        box_count = 0
        while line_count < boxes_max_count:
            box_text = []
            box_text.append(text[line_count])
            line_count += 1
            while text[line_count][0:3] != '## ': # Next box
                box_text.append(text[line_count])
                line_count += 1
                if line_count >= boxes_max_count:
                    break
            box = Box.Box(box_text)
            box.index = box_count
            self.boxes.append(box)
            box_count += 1
        line_count += 1
    
        # Load project
        project_text = []
        while line_count < max_count:
            if text[line_count][0] != '\n':
                project_text.append(text[line_count])
            line_count += 1
        self.project = Project.Project(project_text)
        return

    # Render session object as Markdown or HTML
    def render(self, course_name, type="MD"):
        output = []
        if type == "MD":
            output.append(f"# {course_name} : {self.name}\n")
            output.append(f"{self.description}\n\n")
        elif type == "HTML":
            output.append(f"<h1>{course_name} : {self.name}</h1")
            output.append(f"{self.description}<br>")
        for box in self.boxes:
            output.extend(box.render(type=type))
        return output

    # Load session object from JSON
    def load(self, path):
        with open(path, "r") as file:
            self.from_dict(json.load(file))
        return

    # Store session object in JSON file
    def store(self, path):
        with open(path, "w") as file:
            json.dump(self.to_dict(), file, indent=4)
        return

#FIN