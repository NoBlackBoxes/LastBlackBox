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
    def __init__(self, text=None, dictionary=None):
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
        if text:
            self.parse(text)            # Parse material from "materials.csv" text
        elif dictionary:
            self.from_dict(dictionary)  # Load material from dictionary
        return

    # Convert material object to dictionary
    def to_dict(self):
        dictionary = {
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "quantity": self.quantity,
            "datasheet": self.datasheet,
            "supplier": self.supplier,
            "package": self.package,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "unit_price": self.unit_price,
            "bulk_price": self.bulk_price,
            "new": self.new,
            "used": self.used
        }
        return dictionary

    # Convert dictionary to material object
    def from_dict(self, dictionary):
        self.name = dictionary.get("name")
        self.slug = dictionary.get("slug")
        self.description = dictionary.get("description")
        self.quantity = dictionary.get("quantity")
        self.datasheet = dictionary.get("datasheet")
        self.supplier = dictionary.get("supplier")
        self.package = dictionary.get("package")
        self.x = dictionary.get("x")
        self.y = dictionary.get("y")
        self.z = dictionary.get("z")
        self.unit_price = dictionary.get("unit_price"),
        self.bulk_price = dictionary.get("bulk_price"),
        self.new = dictionary.get("new"),
        self.used = dictionary.get("used")
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

    # Load material object from JSON
    def load(self, path):
        with open(path, "r") as file:
            self.from_dict(json.load(file))
        return

    # Store material object in JSON file
    def store(self, path):
        with open(path, "w") as file:
            json.dump(self.to_dict(), file, indent=4)
        return
#FIN