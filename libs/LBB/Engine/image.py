# -*- coding: utf-8 -*-
"""
LBB : Engine : Image Class

@author: kampff
"""

# Imports
import LBB.config as Config

# Image Class
class Image:
    """
    LBB Image Class

    Stores a link to an image resource
    """ 
    def __init__(self, _lesson, text):
        self.course = _lesson.course        # Image parent (course)
        self.session = _lesson.session      # Image parent (session)
        self.box = _lesson.box              # Image box
        self.name = None                    # Image name
        self.width = None                   # Image width
        self.path = None                    # Image path
        self.url = None                     # Image URL
        self.parse(text)                    # Parse image from Markdown text
        return
        
    # Parse image string
    def parse(self, text):
        name_width = text.split(']')[0][2:]   # strip ![
        if ':' in name_width:
            self.name, self.width = name_width.split(':', 1)
        else:
            self.name = name_width
            self.width = "480"  # default
        self.path = text.split('(')[1][:-1]
        if self.path.startswith("http"):
            self.url = self.path
        else:
            self.url = ("https://raw.githubusercontent.com/NoBlackBoxes/LastBlackBox/master" + self.path)
        return

    # Render image object as Markdown
    def render(self):
        output = []
        image_prefix = Config.image_prefix
        output.append(f"<p align=\"center\">")
        output.append(f"<img src=\"{image_prefix}{self.path}\" alt=\"{self.name}\" width=\"{self.width}\">")
        output.append(f"</p>\n")
        return output

#FIN