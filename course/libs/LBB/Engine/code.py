# -*- coding: utf-8 -*-
"""
LBB: Code Class

@author: kampff
"""

# Import libraries
import re

# Import modules
import LBB.Engine.utilities as Utilities

# Code Class
class Code:
    def __init__(self, text=None, dictionary=None):
        self.index = None               # Step index
        self.type = "code"              # Step type
        self.depth = None               # Step depth
        self.syntax = None              # Code syntax
        self.content = None             # Code content
        if text:
            self.parse(text)            # Parse code from README text
        elif dictionary:
            self.from_dict(dictionary)  # Load instruction from dictionary
        return
        
    # Convert instruction object to dictionary
    def to_dict(self):
        dictionary = {
            "index": self.index,
            "type": self.type,
            "depth": self.depth,
            "syntax": self.syntax,
            "content": self.content
        }
        return dictionary

    # Convert dictionary to instruction object
    def from_dict(self, dictionary):
        self.index = dictionary.get("index")
        self.type = dictionary.get("type")
        self.depth = dictionary.get("depth")
        self.syntax = dictionary.get("syntax")
        self.content = dictionary.get("content")
        return

    # Parse instruction string
    def parse(self, text):
        self.syntax = text[0].split("```")[1].strip()
        self.content = text[1:-1]
        return

    # Render instruction object as Markdown or HTML
    def render(self, type="MD"):
        output = []
        if type == "MD":
            output.append(f"```{self.syntax}\n")
            for index, line in enumerate(self.content):
                if (index > 0) and (line.startswith("#")):
                    output.append(f"\n")
                output.append(f"{line}\n")
            output.append(f"```\n\n")
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