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
        self.html = None                # Image html
        if text:
            self.parse(text)            # Parse image from README text
        elif dictionary:
            self.from_dict(dictionary)  # Load image from dictionary
        return
    
    # Convert image object to dictionary
    def to_dict(self):
        dictionary = {}
        dictionary.update({"index": self.index})
        dictionary.update({"type": self.type})
        dictionary.update({"html": self.html})
        return dictionary

    # Convert dictionary to image object
    def from_dict(self, dictionary):
        self.index = dictionary.get("index")
        self.type = dictionary.get("type")
        self.html = dictionary.get("html")
        return
    
    # Parse image string
    def parse(self, text):
        if not text.split("src=")[1].startswith("http"):
            self.html = text.replace("../../..", "https://raw.githubusercontent.com/NoBlackBoxes/LastBlackBox/master")
        else:
            self.html = text
        return

#FIN