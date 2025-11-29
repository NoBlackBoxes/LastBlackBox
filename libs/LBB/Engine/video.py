# -*- coding: utf-8 -*-
"""
LBB : Engine : Video Class

@author: kampff
"""

# Imports
import LBB.config as Config

# Video Class
class Video:
    """
    LBB Video Class

    Stores a link to a video tutorial
    """ 
    def __init__(self, _lesson, text):
        self.course = _lesson.course        # Video parent (course)
        self.session = _lesson.session      # Video parent (session)
        self.box = _lesson.box              # Video box
        self.name = None                    # Video name
        self.url = None                     # Video url
        self.id = None                      # Video id
        self.parse(text)                    # Parse video from Markdown text
        return
    
    # Parse video string
    def parse(self, text):
        split_line = text.split(']') 
        self.name = split_line[0][1:]
        self.url = split_line[1].split('(')[1][:-1]
        self.id = self.url.split('/')[-1]
        return

    # Render video object as Markdown
    def render(self):
        image_prefix = Config.image_prefix
        gif_name = self.name + ".gif"
        gif_name = gif_name.replace('(', '').replace(')', '').replace('\'', '').replace(',', '')
        if gif_name.startswith("NB3"):
            gif_name = "NB3_" + gif_name[6:]
        gif_name = gif_name.replace(" ", "-")
        output = f"<p align=\"center\">\n<a href=\"{self.url}\" title=\"Control+Click to watch in new tab\"><img src=\"{image_prefix}/boxes/{self.box}/_resources/lessons/thumbnails/{gif_name}\" alt=\"{self.name}\" width=\"480\"/></a>\n</p>\n"
        return output

#FIN
