# -*- coding: utf-8 -*-
"""
LBB: Image Class

@author: kampff
"""

# Import libraries

# Import modules

# Image Class
class Image:
    def __init__(self, text=None, dictionary=None):
        self.index = None               # Step index
        self.type = "image"             # Step type
        self.depth = None               # Step depth
        self.name = None                # Image name
        self.width = None               # Image width
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
        url = text.split('(')[1][:-1]
        if url.startswith("http"):
            self.url = url
        else:
            self.url = url.replace("../..", "https://raw.githubusercontent.com/NoBlackBoxes/LastBlackBox/master")
        return

#FIN