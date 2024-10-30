# -*- coding: utf-8 -*-
"""
LBB: Task Class

@author: kampff
"""

# Import modules
import LBB.utilities as Utilities

# Task Class
class Task:
    def __init__(self, text=None, dictionary=None):
        self.index = None               # Step index
        self.type = "task"              # Step type
        self.description = None         # Task description
        self.result= None               # Task result
        if text:
            self.parse(text)            # Parse task from README text
        elif dictionary:
            self.from_dict(dictionary)  # Load task from dictionary
        return
    
    # Convert task object to dictionary
    def to_dict(self):
        dictionary = {}
        dictionary.update({"index": self.index})
        dictionary.update({"type": self.type})
        dictionary.update({"description": self.description})
        dictionary.update({"result": self.result})
        return dictionary

    # Convert dictionary to task object
    def from_dict(self, dictionary):
        self.index = dictionary.get("index")
        self.type = dictionary.get("type")
        self.description = dictionary.get("description")
        self.result = dictionary.get("result")
        return
    
    # Parse task string
    def parse(self, text):
        self.description = text[0][16:]
        self.description = Utilities.convert_emphasis_tags(self.description)
        self.description = Utilities.convert_markdown_links(self.description)
        self.result = text[1].split("> **Expected Result**: ")[1]
        self.result = Utilities.convert_emphasis_tags(self.result)
        self.result = Utilities.convert_markdown_links(self.result)
        return

#FIN