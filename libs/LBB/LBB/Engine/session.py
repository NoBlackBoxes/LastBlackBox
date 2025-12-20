# -*- coding: utf-8 -*-
"""
LBB : Engine : Session Class

@author: kampff
"""

# Imports
import os
import LBB.config as Config
import LBB.utilities as Utilities
import LBB.Engine.lesson as Lesson
import LBB.Engine.material as Material

# Session Class
class Session:
    """
    LBB Session Class

    Stores a sequence of lessons and the materials required to complete them
    """
    def __init__(self, _course, text):
        self.course = _course           # Session parent (course)
        self.index = None               # Session index
        self.template = None            # Session template text
        self.name = None                # Session name
        self.slug = None                # Session slug (URL)
        self.description = None         # Session description
        self.lessons = None             # Session lessons
        self.materials = None           # Session materials
        self.parse(text)                # Parse session from template text
        return
    
    # Parse session template
    def parse(self, text):
        # Set line counter
        line_count = 0

        # Extract name and slug
        title = text[line_count].strip()
        self.name = title.split(':')[1].strip()
        self.slug = self.name.lower().replace(' ', '-')
        line_count += 1

        # Extract description
        self.description = text[line_count]
        line_count += 1

        # Store template
        self.template = text[line_count:]

        # Find lesson tags
        lesson_tags = Utilities.find_lesson_tags(text)

        # Load lessons
        self.lessons = []
        lesson_boxes = set()
        lesson_materials = []
        for lesson_tag in lesson_tags:
            tag_content = lesson_tag[1:-1].split(":")
            if len(tag_content) != 2:
                print(f"Invalid Lesson Tag: {lesson_tag} in session: {self.name}")
                exit(-1)
            lesson_box = tag_content[0].lower()
            lesson_name = tag_content[1]
            lesson_path = f"{Config.boxes_path}/{lesson_box}/_resources/lessons/{lesson_name}.md"
            lesson_text = Utilities.read_clean_text(lesson_path)
            lesson = Lesson.Lesson(self, lesson_box, lesson_text)
            self.lessons.append(lesson)
            lesson_boxes.add(lesson_box)

            # Does lesson require additional materials?
            lesson_materials_path = f"{lesson_path[:-3]}.csv"
            if os.path.exists(lesson_materials_path):
                lesson_materials_text = Utilities.read_clean_text(lesson_materials_path)
                for material_text in lesson_materials_text:
                    if material_text.startswith("name"):
                        continue
                    material = Material.Material(text=material_text)
                    lesson_materials.append(material)

        # Load core materials from the set of boxes used for this sessions's lessons
        box_materials = []
        for lesson_box in lesson_boxes:
            materials_path = f"{Config.boxes_path}/{lesson_box}/_resources/materials.csv"
            materials_text = Utilities.read_clean_text(materials_path)
            for material_text in materials_text:
                if material_text.startswith("name"):
                    continue
                material = Material.Material(material_text)
                box_materials.append(material)
        
        # Combine materials
        self.materials = box_materials + lesson_materials

        return

    # Render session object as Markdown
    def render(self):
        output = []
        output.append(f"# {self.course.name} : {self.name}\n")
        output.append(f"{self.description}\n\n")

        # Render Materials List
        output.append(f"<details><summary><i>Materials</i></summary><p>\n\n")
        output.append("Name|Description| # |Package|Data|Link|\n")
        output.append(":-------|:----------|:-----:|:-:|:--:|:--:|\n")
        for m in self.materials:
            output.append(f"{m.name}|{m.description}|{m.quantity}|{m.package}|[-D-]({m.datasheet})|[-L-]({m.supplier})\n")
        output.append(f"\n</p></details><hr>\n\n")

        # Render Template (with lesson text insertion)
        lesson_count = 0
        for line in self.template:
            if line.startswith("{"): # Render next lesson
                for lesson_line in self.lessons[lesson_count].render():
                    output.append(lesson_line)
                lesson_count += 1
            else:
                output.append(line + "\n")
        return output

#FIN