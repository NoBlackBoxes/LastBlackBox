# -*- coding: utf-8 -*-
"""
LBB: Session Class

@author: kampff
"""

# Import libraries
import os
import LBB.utilities as Utilities

# Import modules
import LBB.box as Box
import LBB.project as Project

# Session Class
class Session:
    def __init__(self, readme_path=None):
        self.name = None            # session name
        self.description = None     # session description
        self.boxes = None           # session boxes
        self.project = None         # session project
        if readme_path:
            self.parse_readme(readme_path)
        return
    
    def parse_readme(self, readme_path):
        with open(readme_path, encoding='utf8') as f:
            readme = f.readlines()

        # Set line counter
        line_count = 0
        max_count = len(readme)

        # Extract name
        title = readme[line_count][2:-1]
        self.name = title.split('-')[1][1:]
        line_count += 1

        # Extract description
        self.description = []
        while readme[line_count][0] != '#':
            if readme[line_count][0] != '\n':
                self.description.append(readme[line_count][:-1])
            line_count += 1
        self.description = "".join(self.description)

        # Count lines in the boxes section
        box_line_count = line_count
        while readme[box_line_count][0:3] != '---': # End of boxes section
            box_line_count += 1

        # Extract session boxes
        self.boxes = []
        while line_count < box_line_count:
            box_text = []
            box_text.append(readme[line_count][:-1])
            line_count += 1
            while readme[line_count][0:3] != '## ': # Next box
                if readme[line_count][0] != '\n':
                    box_text.append(readme[line_count][:-1])
                line_count += 1
                if line_count >= box_line_count:
                    break
            box = Box.Box(box_text)
            self.boxes.append(box)
        line_count += 2
    
        # Extract session project
        while line_count < max_count:
            project_text = []
            project_text.append(readme[line_count][2:-1])
            line_count += 1
            if readme[line_count][0] != '\n':
                project_text.append(readme[line_count][:-1])
            line_count += 1
        self.project = Project.Project(project_text)
        return

#FIN