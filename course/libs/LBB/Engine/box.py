# -*- coding: utf-8 -*-
"""
LBB: Box Class

@author: kampff
"""

# Import libraries
import json

# Import modules
import LBB.Engine.utilities as Utilities
import LBB.Engine.config as Config
import LBB.Engine.lesson as Lesson
import LBB.Engine.material as Material

# Box Class
class Box:
    """
    LBB Box Class

    Stores a list of lessons and materials required to open this black box
    - Each box has a "depth" (01, 10, 11) indicating the degree of difficulty
    """
    def __init__(self, text=None, dictionary=None):
        self.index = None               # Box index
        self.name = None                # Box name
        self.slug = None                # Box slug (URL)
        self.depth = None               # Box depth
        self.description = None         # Box description
        self.materials = None           # Box materials
        self.lessons = None             # Box lessons
        if text:
            self.parse(text)            # Parse box from template text
        elif dictionary:
            self.from_dict(dictionary)  # Load box from dictionary
        return

    # Convert box object to dictionary
    def to_dict(self):
        dictionary = {
            "index": self.index,
            "name": self.name,
            "slug": self.slug,
            "depth": self.depth,
            "description": self.description,
            "materials": [material.to_dict() for material in self.materials],
            "lessons": [lesson.to_dict() for lesson in self.lessons]
        }
        return dictionary

    # Convert dictionary to box object
    def from_dict(self, dictionary):
        self.index = dictionary.get("index")
        self.name = dictionary.get("name")
        self.slug = dictionary.get("slug")
        self.depth = dictionary.get("depth")
        self.description = dictionary.get("description")
        self.materials = [Material.Material(dictionary=material_dictionary) for material_dictionary in dictionary.get("materials", [])]
        self.lessons = [Lesson.Lesson(dictionary=lesson_dictionary) for lesson_dictionary in dictionary.get("lessons", [])]
        return
    
    # Parse box string
    def parse(self, text):
        # Set line counter
        line_count = 0
        max_count = len(text)
        
        # Extract name, slug, and depth
        title = text[0][2:].strip()
        self.name = title.split("{")[0]
        self.slug = self.name.lower()
        self.depth = title.split("{")[1][:2]
        line_count += 1

        # Load description
        info_path = f"{Config.boxes_root}/{self.slug}/_resources/info.md"
        info_text = Utilities.read_clean_text(info_path)
        description_line = Utilities.find_line(info_text, "## Description") + 1
        self.description = info_text[description_line]

        # List box depths
        depths = Utilities.get_depths(self.depth)

        # Load materials
        materials_path = f"{Config.boxes_root}/{self.slug}/_resources/materials.csv"
        materials_text = Utilities.read_clean_text(materials_path)
        materials = []
        for material_text in materials_text:
            material_depth = material_text.split(",")[1]
            if material_depth in depths:
                material = Material.Material(material_text)
                material.datasheet = f"/boxes/{self.slug}/{material.datasheet}"
                materials.append(material)
        self.materials = materials

        # Load lessons
        self.lessons = []
        lesson_count = 0
        while line_count < max_count:
            if not text[line_count].startswith("{"):
                print(f"Invalid Lesson Tag: {text[line_count]}")
                exit(-1)
            lesson_basename = text[line_count].split("{")[1][:-1]
            lesson_path = f"{Config.boxes_root}/{self.slug}/_resources/lessons/{lesson_basename}.md"
            lesson_text = Utilities.read_clean_text(lesson_path)
            lesson = Lesson.Lesson(self.depth, text=lesson_text)
            lesson.index = lesson_count
            self.lessons.append(lesson)
            lesson_count += 1
            line_count += 1
        return

    # Render box object as Markdown or HTML
    def render(self, type="MD"):
        output = []
        if type == "MD":
            output.append(f"## {self.name}\n")
            output.append(f"{self.description}\n\n")
            output.append(f"<details><summary><i>Materials</i></summary><p>\n\n")
            output.append("Contents|Depth|Description| # |Data|Link|\n")
            output.append(":-------|:---:|:----------|:-:|:--:|:--:|\n")
            for m in self.materials:
                output.append(f"{m.part}|{m.depth}|{m.description}|{m.quantity}|[-D-]({m.datasheet})|[-L-]({m.supplier})\n")
            output.append(f"\n</p></details><hr>\n\n")

        elif type == "HTML":
            output.append(f"<h2>{self.name}</h2")
            output.append(f"{self.description}<br>")
        for lesson in self.lessons:
            for line in lesson.render(type=type):
                output.append(line)
            pass
        return output

#FIN