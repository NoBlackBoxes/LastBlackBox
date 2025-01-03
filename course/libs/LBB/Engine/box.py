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
    - Each box has "depth level" (01, 10, 11) indicating the degree of difficulty
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
        print(text)
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
            print(f"Invalid Depth Level: {self.depth}")
            exit(-1)

        # Load materials
        materials_path = f"{Config.boxes_root}/{self.slug}/_resources/materials.csv"
        materials_text = Utilities.read_clean_text(materials_path)
        materials = []
        for material_text in materials_text:
            material_depth = material_text.split(",")[1]
            if material_depth in depths:
                material = Material.Material(material_text)
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

            print(lesson_path)

            lesson_text = Utilities.read_clean_text(lesson_path)
            lesson = Lesson.Lesson(lesson_text)
            lesson.index = lesson_count
            self.lessons.append(lesson)
            lesson_count += 1
            line_count += 1
        return

    # Render box object as Markdown or HTML
    def render(self):
        output = []
        output.append(f"## {self.name}\n")
        output.append(f"{self.description}\n")
        for lesson in self.lessons:
            output.append(lesson.render())
        return output

    # Load box object from JSON
    def load(self, path):
        with open(path, "r") as file:
            self.from_dict(json.load(file))
        return

    # Store box object in JSON file
    def store(self, path):
        with open(path, "w") as file:
            json.dump(self.to_dict(), file, indent=4)
        return
#FIN