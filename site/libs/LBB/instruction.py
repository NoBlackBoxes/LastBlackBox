# -*- coding: utf-8 -*-
"""
LBB: Instruction Class

@author: kampff
"""

# Import libraries
import re

# Import modules

# Instruction Class
class Instruction:
    def __init__(self, text=None):
        self.html = None     # instruction text
        if text:
            self.parse(text)
        return
    
    def convert_emphasis_tags(self, text):
        text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', text)
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
        return text

    def parse(self, text):
        text = self.convert_emphasis_tags(text)
        if text[0] == '>': # Block Quote
            self.html = f"<blockquote id=\"quote\">{text[1:].strip()}</blockquote>"
        else:
            self.html = f"<h4 id=\"instruction\">{text}</h4>\n"

        # TO DO: convert emphasis tags
        return

    def render(self):
        return self.html
#FIN