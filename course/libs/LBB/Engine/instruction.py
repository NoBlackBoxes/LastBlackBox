# -*- coding: utf-8 -*-
"""
LBB: Instruction Class

@author: kampff
"""

# Import libraries
import re

# Import modules
import LBB.Engine.utilities as Utilities

# Instruction Class
class Instruction:
    def __init__(self, text=None, dictionary=None):
        self.index = None               # Step index
        self.type = "instruction"       # Step type
        self.depth = None               # Step depth
        self.content = None             # Instruction content
        if text:
            self.parse(text)            # Parse instruction from README text
        elif dictionary:
            self.from_dict(dictionary)  # Load instruction from dictionary
        return
        
    # Convert instruction object to dictionary
    def to_dict(self):
        dictionary = {
            "index": self.index,
            "type": self.type,
            "depth": self.depth,
            "html": self.html
        }
        return dictionary

    # Convert dictionary to instruction object
    def from_dict(self, dictionary):
        self.index = dictionary.get("index")
        self.type = dictionary.get("type")
        self.html = dictionary.get("html")
        return

    # Parse instruction string
    def parse(self, text):
        self.content = text
        return

    # Render instruction object as Markdown or HTML
    def render(self, type="MD"):
        output = []
        if type == "MD":
            output.append(f"- {self.content}\n")
        elif type == "HTML":
            text = self.content
            text = Utilities.convert_emphasis_tags(text)
            text = Utilities.convert_markdown_links(text)
            if text[0] == '>': # Block Quote
                text = f"<blockquote id=\"quote\">{text[1:].strip()}</blockquote>\n"
            else:
                text = f"<span id=\"instruction\">{text}</span>\n"
            output.append(text)
        return output

#FIN