# -*- coding: utf-8 -*-
"""
LBB : Engine : Box Class

@author: kampff
"""

# Imports
import os
import json
import LBB.utilities as Utilities
import LBB.config as Config
import LBB.Engine.lesson as Lesson
import LBB.Engine.material as Material

# Box Class
class Box:
    """
    LBB Box Class

    Stores a list of lessons and materials required to open this black box
    """
    def __init__(self, _session, text=None, dictionary=None):
        self.course = _session.course   # Box parent (course)
        self.session = _session         # Box parent (session)
        self.index = None               # Box index
        self.name = None                # Box name
        self.slug = None                # Box slug (URL)
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
        self.description = dictionary.get("description")
        self.materials = [Material.Material(dictionary=material_dictionary) for material_dictionary in dictionary.get("materials", [])]
        self.lessons = [Lesson.Lesson(self, dictionary=lesson_dictionary) for lesson_dictionary in dictionary.get("lessons", [])]
        return
    
    # Parse box string
    def parse(self, text):
        # Set line counter
        line_count = 0
        max_count = len(text)
        
        # Extract name and slug
        title = text[0][2:].strip()
        self.name = title.split("{")[0]
        self.slug = self.name.lower()
        line_count += 1

        # Load description
        info_path = f"{Config.boxes_path}/{self.slug}/_resources/info.md"
        info_text = Utilities.read_clean_text(info_path)
        description_line = Utilities.find_line(info_text, "## Description") + 1
        self.description = info_text[description_line]

        # Load materials
        materials_path = f"{Config.boxes_path}/{self.slug}/_resources/materials.csv"
        materials_text = Utilities.read_clean_text(materials_path)
        materials = []
        for material_text in materials_text:
            if material_text.startswith("name"):
                continue
            material = Material.Material(text=material_text)
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
            lesson_path = f"{Config.boxes_path}/{self.slug}/_resources/lessons/{lesson_basename}.md"
            lesson_text = Utilities.read_clean_text(lesson_path)
            lesson = Lesson.Lesson(self, text=lesson_text)
            lesson.index = lesson_count
            self.lessons.append(lesson)
            # Does lesson require additional materials?
            lesson_materials_path = f"{lesson_path[:-3]}.csv"
            if os.path.exists(lesson_materials_path):
                lesson_materials_text = Utilities.read_clean_text(lesson_materials_path)
                for material_text in lesson_materials_text:
                    if material_text.startswith("name"):
                        continue
                    material = Material.Material(text=material_text)
                    self.materials.append(material)
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
            output.append("Name|Description| # |Package|Data|Link|\n")
            output.append(":-------|:----------|:-----:|:-:|:--:|:--:|\n")
            for m in self.materials:
                output.append(f"{m.name}|{m.description}|{m.quantity}|{m.package}|[-D-]({m.datasheet})|[-L-]({m.supplier})\n")
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