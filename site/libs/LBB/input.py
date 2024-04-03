# -*- coding: utf-8 -*-
"""
LBB: Input Class

@author: kampff
"""

# Import libraries
import os

# Import modules

# Input Class
class Input:
    def __init__(self, _type):
        self.type = _type        # type
        self.value = None        # value
        
    def render(self):
        output = []
        if self.type == "number":
            output.append("<input type=\"text\" id=\"test\" name = \"test\">")
        if self.type == "photo":
            output.append("<input type=\"file\" id=\"picture\" name = \"picture\" accept=\"image/x-png,image/jpeg,image/gif\">")
        return "".join(output)

#FIN