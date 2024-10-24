# -*- coding: utf-8 -*-
"""
LBB: Box Class

@author: kampff
"""

# Import libraries

# Import modules
import LBB.lesson as Lesson

# Box Class
class Box:
    def __init__(self, text=None):
        self.name = None            # box name
        self.description = None     # box description
        self.lessons = None         # box lessons
        if text:
            self.parse(text)
        return

    def parse(self, text):
        # Set line counter
        line_count = 0
        max_count = len(text)

         # Extract name
        self.name = text[0][3:]
        line_count += 1

        # Extract description
        self.description = []
        while text[line_count][0] != '#':
            if text[line_count][0] != '\n':
                self.description.append(text[line_count][:-1])
            line_count += 1
        self.description = "".join(self.description)

        # Extract lessons
        self.lessons = []
        while line_count < max_count:
            lesson_text = []
            lesson_text.append(text[line_count])
            line_count += 1
            while text[line_count][0:9] != '### Watch': # Next lesson
                if text[line_count][0] != '\n':
                    lesson_text.append(text[line_count])
                line_count += 1
                if line_count >= max_count:
                    break
            lesson = Lesson.Lesson(lesson_text)
            self.lessons.append(lesson)

    def render(self):
        output = ''
        #for lesson in self.lessons:
        #    output = output + lesson.render()
        return output
#FIN