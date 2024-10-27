# -*- coding: utf-8 -*-
"""
LBB: Video Class

@author: kampff
"""

# Import libraries

# Import modules

# Video Class
class Video:
    def __init__(self, text=None, dictionary=None):
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
        dictionary = {}
        dictionary.update({"name": self.name})
        dictionary.update({"url": self.url})
        dictionary.update({"id": self.id})
        return dictionary

    # Convert dictionary to video object
    def from_dict(self, dictionary):
        self.name = dictionary.get("name")
        self.url = dictionary.get("url")
        self.id = dictionary.get("id")
        return

    # Parse video string
    def parse(self, text):
        split_string = text.split('[')[1].split(']')
        self.name = split_string[0]
        self.url = split_string[1][1:-1]
        self.id = self.url.split('/')[-1]
        return

    def render(self):
        embed_string = f"<iframe id=\"video_player\" src=\"https://player.vimeo.com/video/{self.id}?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479\" width=\"640\" height=\"360\" frameborder=\"0\" allow=\"autoplay; fullscreen; picture-in-picture; clipboard-write\" title=\"{self.name}\"></iframe>"
        return "<p align=\"center\">\n" + embed_string + "\n</p>\n"

#FIN
