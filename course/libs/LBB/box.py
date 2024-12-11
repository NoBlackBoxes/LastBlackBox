# -*- coding: utf-8 -*-
"""
LBB: Box Class

@author: kampff
"""

# Import libraries
import json

# Import modules
import LBB.config as Config
import LBB.lesson as Lesson
import LBB.material as Material

# Box Class
class Box:
    def __init__(self, text=None, dictionary=None):
        self.index = None               # Box index
        self.name = None                # Box name
        self.slug = None                # Box slug (URL)
        self.depth = None               # Box depth
        self.description = None         # Box description
        self.lessons = None             # Box lessons
        self.materials = None           # Box materials
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
            "lessons": [lesson.to_dict() for lesson in self.lessons],
            "materials": [material.to_dict() for material in self.materials]
        }
        return dictionary

    # Convert dictionary to box object
    def from_dict(self, dictionary):
        self.index = dictionary.get("index")
        self.name = dictionary.get("name")
        self.slug = dictionary.get("slug")
        self.depth = dictionary.get("depth")
        self.description = dictionary.get("description")
        self.lessons = [Lesson.Lesson(dictionary=lesson_dictionary) for lesson_dictionary in dictionary.get("lessons", [])]
        self.materials = [Material.Material(dictionary=material_dictionary) for material_dictionary in dictionary.get("materials", [])]
        return
    
    # Parse box string
    def parse(self, text):
        # Set line counter
        line_count = 0
        max_count = len(text)

        # Extract name and depth
        title = text[0][2:].strip()
        self.name = title.split("(")[0]
        self.slug = self.name.lower()
        self.depth = title.split("(")[1][:2]
        line_count += 1

        # Extract description
        self.description = text[line_count]
        line_count += 1

        # List box depths
        depths = []
        if self.depth == "01":
            depths.append("01")
        if self.depth == "10":
            depths.append("01")
            depths.append("10")
        if self.depth == "11":
            depths.append("01")
            depths.append("10")
            depths.append("11")

        # Load materials
        materials_path = f"{Config.boxes_root}/{self.slug}/materials.csv"
        with open(materials_path, encoding='utf8') as f:
            materials_text = f.readlines()
        materials = []
        for material_text in materials_text:
            material_depth = material_text.split(",")[1]
            if material_depth in depths:
                material = Material.Material(material_text)
                materials.append(material)
        print(materials)

        ## Extract lessons
        #self.lessons = []
        #lesson_count = 0
        #while line_count < max_count:
        #    lesson_text = []
        #    lesson_text.append(text[line_count])         # Append lesson heading
        #    line_count += 1
        #    while not text[line_count].startswith('##'): # Next lesson
        #        lesson_text.append(text[line_count])
        #        line_count += 1
        #        if line_count >= max_count:
        #            break
        #    lesson = Lesson.Lesson(lesson_text)
        #    lesson.index = lesson_count
        #    self.lessons.append(lesson)
        #    lesson_count += 1
        return

    def render(self):
        output = ''
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