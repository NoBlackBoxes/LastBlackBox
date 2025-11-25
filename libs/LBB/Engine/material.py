# -*- coding: utf-8 -*-
"""
LBB : Engine : Material Class

@author: kampff
"""

# Imports
import json
import re

# Material Class
class Material:
    """
    LBB Material Class

    Describes a hardware material used to open a box in LBB
    """
    def __init__(self, text):
        self.name = None                # Material name
        self.slug = None                # Material slug
        self.description = None         # Material description
        self.quantity = None            # Material quantity
        self.datasheet = None           # Material datasheet (link)
        self.supplier = None            # Material supplier (link)
        self.package = None             # Material package name
        self.x = None                   # Material x dimension (mm)
        self.y = None                   # Material y dimension (mm)
        self.z = None                   # Material z dimension (mm)
        self.unit_price = None          # Material unit price (£)
        self.bulk_price = None          # Material bulk price (£)
        self.new = None                 # Material new stock
        self.used = None                # Material used stock
        self.parse(text)                # Parse material from "materials.csv" text
        return
    
    # Parse material string
    def parse(self, text):
        fields = text.split(",")
        self.name = fields[0].strip()
        self.slug = fields[1].strip()
        self.description = fields[2].strip()
        self.quantity = int(fields[3].strip())
        self.datasheet = fields[4].strip()
        self.supplier = fields[5].strip()
        self.package = fields[6].strip()
        self.x = int(fields[7].strip())
        self.y = int(fields[8].strip())
        self.z = int(fields[9].strip())
        self.unit_price = fields[10].strip()
        self.bulk_price = fields[11].strip()
        self.new = fields[12].strip()
        self.used = fields[13].strip()
        return

#FIN