# -*- coding: utf-8 -*-
"""
LBB Site Site Site: Box Class

@author: kampff
"""

# Import modules
import Site.lesson as Lesson

# Box Class
class Box:
    def __init__(self, text=None, dictionary=None):
        self.index = None               # Box index
        self.name = None                # Box name
        self.slug = None                # Box slug (URL)
        self.description = None         # Box description
        self.lessons = None             # Box lessons
        if text:
            self.parse(text)            # Parse box from README text
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
            "lessons": [lesson.to_dict() for lesson in self.lessons]
        }
        return dictionary

    # Convert dictionary to box object
    def from_dict(self, dictionary):
        self.index = dictionary.get("index")
        self.name = dictionary.get("name")
        self.slug = dictionary.get("slug")
        self.description = dictionary.get("description")
        self.lessons = [Lesson.Lesson(dictionary=lesson_dictionary) for lesson_dictionary in dictionary.get("lessons", [])]
        return
    
    # Parse box string
    def parse(self, text):
        # Set line counter
        line_count = 0
        max_count = len(text)

         # Extract name
        self.name = text[0][3:]
        self.slug = self.name.lower().replace(' ', '-')
        line_count += 1

        # Extract description
        self.description = []
        while text[line_count][0] != '#':
            if text[line_count][0] != '\n':
                self.description.append(text[line_count])
            line_count += 1
        self.description = "".join(self.description)

        # Extract lessons
        self.lessons = []
        lesson_count = 0
        while line_count < max_count:
            lesson_text = []
            lesson_text.append(text[line_count])
            line_count += 1
            while not text[line_count].startswith('#### Watch'): # Next lesson
                if text[line_count][0] != '\n':
                    lesson_text.append(text[line_count])
                line_count += 1
                if line_count >= max_count:
                    break
            lesson = Lesson.Lesson(lesson_text)
            lesson.index = lesson_count
            self.lessons.append(lesson)
            lesson_count += 1
        return

    def render(self):
        output = ''
        #for lesson in self.lessons:
        #    output = output + lesson.render()
        return output
#FIN