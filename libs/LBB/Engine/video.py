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
        output = ''
        if type == "MD":
            output = f"[{self.name}]({self.url})"
        elif type == "HTML":
            embed_string = f"<iframe id=\"video_player\" src=\"https://player.vimeo.com/video/{self.id}?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479\" width=\"640\" height=\"360\" frameborder=\"0\" allow=\"autoplay; fullscreen; picture-in-picture; clipboard-write\" title=\"{self.name}\"></iframe>"
            output = f"<p align=\"center\">\n{embed_string}\n</p>\n"
        return output

#FIN
