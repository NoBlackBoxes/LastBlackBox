# -*- coding: utf-8 -*-
"""
LBB: Material Class

@author: kampff
"""

# Import libraries
import json

# Import modules

# Material Class
class Material:
    """
    LBB Material Class

    Describes a hardware material used to open a box in LBB
    """
    def __init__(self, text=None, dictionary=None):
        self.part = None                # Material part (name)
        self.depth = None               # Material depth
        self.description = None         # Material description
        self.quantity = None            # Material quantity
        self.datasheet = None           # Material datasheet (link)
        self.supplier = None            # Material supplier (link)
        self.package = None             # Material package name
        self.x = None                   # Material x dimension (mm)
        self.y = None                   # Material y dimension (mm)
        self.z = None                   # Material z dimension (mm)
        if text:
            self.parse(text)            # Parse material from "materials.csv" text
        elif dictionary:
            self.from_dict(dictionary)  # Load material from dictionary
        return

    # Convert material object to dictionary
    def to_dict(self):
        dictionary = {
            "part": self.part,
            "depth": self.depth,
            "description": self.description,
            "quantity": self.quantity,
            "datasheet": self.datasheet,
            "supplier": self.supplier,
            "package": self.package,
            "x": self.x,
            "y": self.y,
            "z": self.z
        }
        return dictionary

    # Convert dictionary to material object
    def from_dict(self, dictionary):
        self.part = dictionary.get("part")
        self.depth = dictionary.get("depth")
        self.description = dictionary.get("description")
        self.quantity = dictionary.get("quantity")
        self.datasheet = dictionary.get("datasheet")
        self.supplier = dictionary.get("supplier")
        self.package = dictionary.get("package")
        self.x = dictionary.get("x")
        self.y = dictionary.get("y")
        self.z = dictionary.get("z")
        return
    
    # Parse material string
    def parse(self, text):
        fields = text.split(",")
        self.part = fields[0].strip()
        self.depth = fields[1].strip()
        self.description = fields[2].strip()
        self.quantity = int(fields[3].strip())
        self.datasheet = fields[4].strip()
        self.supplier = fields[5].strip()
        self.package = fields[6].strip()
        self.x = int(fields[7].strip())
        self.y = int(fields[8].strip())
        self.z = int(fields[9].strip())
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