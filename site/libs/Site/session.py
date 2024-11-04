# -*- coding: utf-8 -*-
"""
LBB: Session Class

@author: kampff
"""

# Import libraries

# Import modules
import Site.box as Box
import Site.project as Project

# Session Class
class Session:
    def __init__(self, text=None, dictionary=None):
        self.index = None               # Session index
        self.name = None                # Session name
        self.slug = None                # Session slug (URL)
        self.description = None         # Session description
        self.boxes = None               # Session boxes
        self.project = None             # Session project
        if text:
            self.parse(text)            # Parse session from README text
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

    # Parse session string
    def parse(self, text):
        # Set line counter
        line_count = 0
        max_count = len(text)

        # Extract name and slug
        title = text[line_count][2:]
        self.name = title.split('-')[1][1:]
        self.slug = self.name.lower().replace(' ', '-')
        line_count += 1

        # Extract description
        self.description = []
        while text[line_count][0] != '#':
            if text[line_count][0] != '\n':
                self.description.append(text[line_count])
            line_count += 1
        self.description = "".join(self.description)

        # Count lines in the boxes section
        boxes_line_count = line_count
        while text[boxes_line_count][0:3] != '---': # End of boxes section
            boxes_line_count += 1

        # Extract session boxes
        self.boxes = []
        box_count = 0
        while line_count < boxes_line_count:
            box_text = []
            box_text.append(text[line_count])
            line_count += 1
            while text[line_count][0:3] != '## ': # Next box
                if text[line_count][0] != '\n':
                    box_text.append(text[line_count])
                line_count += 1
                if line_count >= boxes_line_count:
                    break
            box = Box.Box(box_text)
            box.index = box_count
            self.boxes.append(box)
            box_count += 1
        line_count += 2
    
        # Extract session project
        project_text = []
        while line_count < max_count:
            if text[line_count][0] != '\n':
                project_text.append(text[line_count])
            line_count += 1
        self.project = Project.Project(project_text)
        return

#FIN