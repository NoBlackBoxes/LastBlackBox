# -*- coding: utf-8 -*-
"""
LBB: Topic Class

@author: kampff
"""

# Import libraries

# Import modules
import LBB.lesson as Lesson

# Topic Class
class Topic:
    def __init__(self, text=None):
        self.name = None            # topic name
        self.lessons = None         # topic lessons
        if text:
            self.parse(text)
        return

    def parse(self, text):
        # Line counter
        line_count = 0
        max_count = len(text)

         # Extract name
        self.name = text[0][:-1]
        
        # Extract lessons
        self.lessons = []
        line_count = 1
        max_count = len(text)
        while line_count < max_count:
            if text[line_count][0] != '\n':
                line_count += 1
        return

    def render(self):
        output = ''
        for lesson in self.lessons:
            output = output + lesson.render()
        return output
#FIN