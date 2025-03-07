# -*- coding: utf-8 -*-
"""
LBB : Engine : Image Class

@author: kampff
"""

# Imports
import LBB.config as Config

# Image Class
class Image:
    def __init__(self, _course, text=None, dictionary=None):
        self.course = _course           # Image parent (course)
        self.index = None               # Step index
        self.type = "image"             # Step type
        self.depth = None               # Step depth
        self.name = None                # Image name
        self.width = None               # Image width
        self.path = None                # Image path
        self.url = None                 # Image URL
        if text:
            self.parse(text)            # Parse image from README text
        elif dictionary:
            self.from_dict(dictionary)  # Load image from dictionary
        return
    
    # Convert image object to dictionary
    def to_dict(self):
        dictionary = {
            "index": self.index,
            "type": self.type,
            "depth": self.depth,
            "name": self.name,
            "width": self.width,
            "url": self.url
        }
        return dictionary

    # Convert dictionary to image object
    def from_dict(self, dictionary):
        self.index = dictionary.get("index")
        self.type = dictionary.get("type")
        self.depth = dictionary.get("depth")
        self.name = dictionary.get("name")
        self.width = dictionary.get("width")
        self.url = dictionary.get("url")
        return
    
    # Parse image string
    def parse(self, text):
        name_width = text.split(']')[0][2:]
        self.name = name_width.split(':')[0]
        self.width = name_width.split(':')[1]
        self.path = text.split('(')[1][:-1]
        if self.path.startswith("http"):
            self.url = self.path
        else:
            self.url = "https://raw.githubusercontent.com/NoBlackBoxes/LastBlackBox/master" + self.path
        return

    # Render image object as Markdown or HTML
    def render(self, type="MD"):
        output = []
        if type == "MD":
            output.append(f"<p align=\"center\">\n")
            output.append(f"<img src=\"{self.course.image_prefix}{self.path}\" alt=\"{self.name}\" width=\"{self.width}\">\n")
            output.append(f"</p>\n\n")
        elif type == "HTML":
            output.append(f"<p align=\"center\">\n")
            output.append(f"<img src=\"{self.url}\" alt=\"{self.name}\" width=\"{self.width}\">\n")
            output.append(f"</p>\n")
        return output

#FIN