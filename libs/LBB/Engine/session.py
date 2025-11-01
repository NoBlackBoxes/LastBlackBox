# -*- coding: utf-8 -*-
"""
LBB : Engine : Session Class

@author: kampff
"""

# Imports
import os
import json
import LBB.config as Config
import LBB.utilities as Utilities
import LBB.Engine.box as Box
import LBB.Engine.project as Project
import LBB.Engine.material as Material

# Session Class
class Session:
    """
    LBB Session Class

    Stores a list of boxes opened during this session and the session project
    """
    def __init__(self, _course, text=None, dictionary=None):
        self.course = _course           # Session parent (course)
        self.index = None               # Session index
        self.name = None                # Session name
        self.slug = None                # Session slug (URL)
        self.description = None         # Session description
        self.boxes = None               # Session boxes
        self.projects = None            # Session projects
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
            "projects": self.projects.to_dict()
        }
        return dictionary

    # Convert dictionary to session object
    def from_dict(self, dictionary):
        self.index = dictionary.get("index")
        self.name = dictionary.get("name")
        self.slug = dictionary.get("slug")
        self.description = dictionary.get("description")
        self.boxes = [Box.Box(self, dictionary=box_dictionary) for box_dictionary in dictionary.get("boxes", [])]
        self.projects = [Project.Project(self, dictionary=project_dictionary) for project_dictionary in dictionary.get("projects", [])]
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
        project_line_count = Utilities.find_line(text, "# Project")

        # Load boxes
        self.boxes = []
        box_count = 0
        while line_count < project_line_count:
            box_text = []
            box_text.append(text[line_count])
            line_count += 1
            while text[line_count][0:3] != '## ': # Next box
                box_text.append(text[line_count])
                line_count += 1
                if line_count >= project_line_count:
                    break
            box = Box.Box(self, text=box_text)
            box.index = box_count
            self.boxes.append(box)
            box_count += 1
    
        # Load project(s)
        line_count = Utilities.find_line(text, "# Project")
        line_count += 1
        self.projects = []
        project_count = 0
        while line_count < max_count:
            if not text[line_count].startswith("{"):
                print(f"Invalid Project Tag: {text[line_count]} - session({self.name})")
                exit(-1)
            project_basename = text[line_count].split("}")[0][1:]
            if len(project_basename.split(":")) != 2:
                print(f"Missing box/lesson name in Project tag - session({self.name})")
                exit(-1)
            project_box_name = project_basename.split(":")[0].lower()
            project_box = next((b for b in self.boxes if b.slug == project_box_name), None)
            project_lesson = project_basename.split(":")[1]
            project_path = f"{Config.boxes_root}/{project_box.slug}/_resources/lessons/{project_lesson}.md"
            project_text = Utilities.read_clean_text(project_path)
            project = Project.Project(project_box, text=project_text)
            project.index = project_count
            self.projects.append(project)
            # Does project require additional materials?
            project_materials_path = f"{project_path[:-3]}.csv"
            if os.path.exists(project_materials_path):
                print(project_materials_path)
                project_materials_text = Utilities.read_clean_text(project_materials_path)
                for material_text in project_materials_text:
                    if material_text.startswith("name"):
                        continue
                    material = Material.Material(text=material_text)
                    project_box.materials.append(material)
            project_count += 1
            line_count += 1
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
            for line in box.render(type=type):
                output.append(line)
        output.append(f"# Project\n")
        for project in self.projects:
            for line in project.render(type=type):
                output.append(line)
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