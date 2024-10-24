# -*- coding: utf-8 -*-
"""
LBB: Video Class

@author: kampff
"""

# Import libraries

# Import modules

# Video Class
class Video:
    def __init__(self, text=None):
        self.name = None        # name
        self.path = None        # path/url
        self.id = None          # id
        if text:
            self.parse(text)
        return
    
    def parse(self, text):
        split_string = text.split('[')[1].split(']')
        self.name = split_string[0]
        self.path = split_string[1][1:-1]
        self.id = self.path.split('/')[-1]
        print(split_string)
        return

    def render(self):
        embed_string = f"<iframe id=\"video_player\" src=\"https://player.vimeo.com/video/{self.id}?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479\" width=\"640\" height=\"360\" frameborder=\"0\" allow=\"autoplay; fullscreen; picture-in-picture; clipboard-write\" title=\"{self.name}\"></iframe>"
        return "<p align=\"center\">\n" + embed_string + "\n</p>\n"

#FIN
