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
    def __init__(self, _course, _box_name, text=None, dictionary=None):
        self.course = _course           # Video parent (course)
        self.box_name = _box_name       # Video box name
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
            "box_name": self.box_name,
            "url": self.url,
            "id": self.id
        }
        return dictionary

    # Convert dictionary to video object
    def from_dict(self, dictionary):
        self.name = dictionary.get("name")
        self.box_name = dictionary.get("box_name")
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
        gif_name = self.name + ".gif"
        if gif_name.startswith("NB3"):
            gif_name = "NB3_" + gif_name[6:]
        gif_name = gif_name.replace(" ", "-")
        output = ''
        if type == "MD":
            output = f"<p align=\"center\">\n<a href=\"{self.url}\" title=\"Control+Click to watch in new tab\"><img src=\"{self.course.image_prefix}/boxes/{self.box_name}/_resources/lessons/thumbnails/{gif_name}\" alt=\"{self.name}\" width=\"480\"/></a>\n</p>\n"
        elif type == "HTML":
            embed_string = f"<iframe id=\"video_player\" src=\"https://player.vimeo.com/video/{self.id}?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479\" width=\"640\" height=\"360\" frameborder=\"0\" allow=\"autoplay; fullscreen; picture-in-picture; clipboard-write\" title=\"{self.name}\"></iframe>"
            output = f"<p align=\"center\">\n{embed_string}\n</p>\n"
        return output

#FIN
