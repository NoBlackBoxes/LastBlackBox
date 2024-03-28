# -*- coding: utf-8 -*-
"""
LBB: Topic Class

@author: kampff
"""

# Import libraries
import os

# Import modules
import LBB.lesson as Lesson

# Topic Class
class Topic:
    def __init__(self, text=None):
        self.name = None            # name
        self.description = None     # description
        self.lessons = None         # lessons
        self.levels = None          # levels
        if text:
            self.parse(text)
        return

    def parse(self, text):
        # Line counter
        line_count = 0
        max_count = len(text)

         # Extract name
        self.name = text[0]

        # Extract description
        self.description = []
        line_count = 1
        while text[line_count][0] != '{':
            if text[line_count][0] != '\n':
                self.description.append(text[line_count])
            line_count += 1
        self.description = "".join(self.description)
        
        # Extract lessons
        self.lessons = []
        while line_count < max_count:
            lesson_text = []
            lesson_text.append(text[line_count])
            line_count += 1
            while text[line_count][0] != '{':
                if text[line_count][0] != '\n':
                    lesson_text.append(text[line_count])
                line_count += 1
                if line_count >= max_count:
                    break
            lesson = Lesson.Lesson(lesson_text)
            self.lessons.append(lesson)
        
        # Extract levels
        self.levels = [lesson.level for lesson in self.lessons]

    def render(self):
        output = ''
        for lesson in self.lessons:
            output = output + lesson.render()
        return output
#FIN