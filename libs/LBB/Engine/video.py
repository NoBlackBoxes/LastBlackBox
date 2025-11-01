# -*- coding: utf-8 -*-
"""
LBB : Engine : Video Class

@author: kampff
"""

# Imports

# Video Class
class Video:
    """
    LBB Video Class

    Stores a link to a video tutorial
    """ 
    def __init__(self, _box, text=None, dictionary=None):
        self.course = _box.course       # Video parent (course)
        self.session = _box.session     # Video parent (session)
        self.box = _box                 # Video parent (box)
        self.name = None                # Video name
        self.url = None                 # Video url
        self.id = None                  # Video id
        if text:
            self.parse(text)            # Parse video from README text
        elif dictionary:
            self.from_dict(dictionary)  # Load lesson from dictionary
        return
    
    # Convert video object to dictionary
    def to_dict(self):
        dictionary = {
            "name": self.name,
            "url": self.url,
            "id": self.id
        }
        return dictionary

    # Convert dictionary to video object
    def from_dict(self, dictionary):
        self.name = dictionary.get("name")
        self.url = dictionary.get("url")
        self.id = dictionary.get("id")
        return

    # Parse video string
    def parse(self, text):
        split_line = text.split(']') 
        self.name = split_line[0][1:]
        self.url = split_line[1].split('(')[1][:-1]
        self.id = self.url.split('/')[-1]
        return

    # Render video in Markdown or HTML
    def render(self, type):
        image_prefix = "../../../.."
        gif_name = self.name + ".gif"
        gif_name = gif_name.replace('(', '').replace(')', '').replace('\'', '')
        if gif_name.startswith("NB3"):
            gif_name = "NB3_" + gif_name[6:]
        gif_name = gif_name.replace(" ", "-")
        output = f"<p align=\"center\">\n<a href=\"{self.url}\" title=\"Control+Click to watch in new tab\"><img src=\"{image_prefix}/boxes/{self.box.slug}/_resources/lessons/thumbnails/{gif_name}\" alt=\"{self.name}\" width=\"480\"/></a>\n</p>\n"
        return output

#FIN
