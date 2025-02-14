# -*- coding: utf-8 -*-
"""
LBB : Packaging : Package Class

@author: kampff
"""

# Import libraries
import random
import cadquery as cq

# Import modules
import LBB.Engine.config as Config
import LBB.Design.layout as Layout
import LBB.Design.svg as SVG
import LBB.Design.png as PNG

# Package Class
class Package:
    """
    LBB : Packaging : Package Class
    """
    def __init__(self, _name, _style, _dimension_type, _l, _w, _h, _material, _wall, _tolerance, rounding_type):
        self.name = _name               # Package name
        self.slug = _name.lower()       # Package slug
        self.style = _style             # Package style
        self.internal = None            # Package internal dimensions (L,W,H or X,Y,Z)
        self.external = None            # Package external dimensions (L,W,H or X,Y,Z)
        self.material = _material       # Package material
        self.wall = _wall               # Package wall thickness
        self.tolerance = _tolerance     # Package tolerance (buffer)
        # Adjust for tolerances
        l = _l - (2*_tolerance)
        w = _w - (2*_tolerance)
        h = _h - (2*_tolerance)
        if _dimension_type == "internal":
            self.internal = {"length":l, "width":w, "height":h}
            self.external = {"length":l+(2*_wall), "width":w+(2*_wall), "height":h+(2*_wall)}
        elif _dimension_type == "external":
            self.external = {"length":l, "width":w, "height":h}
            self.internal = {"length":l-(2*_wall), "width":w-(2*_wall), "height":h-(2*_wall)}
        else:
            print("Invalid package dimension type")
            exit(-1)
        # Round to nearest integer, round down or up
        if rounding_type == "nearest":
            self.internal["length"] = round(self.internal["length"])
            self.internal["width"]  = round(self.internal["width"])
            self.internal["height"] = round(self.internal["height"])
            self.external["length"] = round(self.external["length"])
            self.external["width"]  = round(self.external["width"])
            self.external["height"] = round(self.external["height"])
        elif rounding_type == "down":
            self.internal["length"] = int(self.internal["length"])
            self.internal["width"]  = int(self.internal["width"])
            self.internal["height"] = int(self.internal["height"])
            self.external["length"] = int(self.external["length"])
            self.external["width"]  = int(self.external["width"])
            self.external["height"] = int(self.external["height"])

    def print_dimensions(self):
        description = []
        description.append(f"{self.name}\n")
        description.append("-" * len(self.name) + "\n")
        description.append(f" Style: {self.style}\n")
        description.append(f" Material: {self.material} (wall thickness = {self.wall})\n")
        description.append(f" - Inner: {self.internal["length"]} x {self.internal["width"]} x {self.internal["height"]} mm\n")
        description.append(f" - Outer: {self.external["length"]} x {self.external["width"]} x {self.external["height"]} mm\n")
        description.append(f"\n")
        return "".join(description)

    def generate_model(self):
        outer =cq.Workplane("XY").box(self.external["length"], self.external["width"], self.external["height"])
        z_offset = self.external["height"] - self.internal["height"]
        inner = (
            cq.Workplane("XY")
            .box(self.internal["length"], self.internal["width"], self.internal["height"])
            .translate((0, 0, z_offset))  # Move up to avoid cutting bottom
        )       
        model = outer.cut(inner)
        return model

    def store_designs(self, unit, scale, spacing_ratio, design_folder):
        # Unit dimensions
        num_cols = int(round(self.external["length"] / unit))
        num_rows = int(round(self.external["width"] / unit))
        num_deps = int(round(self.external["height"] / unit))

        # Top/Bottom (l x w)
        store_design(self, "Top", num_rows, num_cols, spacing_ratio, Config.box_names, scale, design_folder)
        store_design(self, "Bottom", num_rows, num_cols, spacing_ratio, Config.box_names, scale, design_folder)
        store_design(self, "Front", num_deps, num_cols, spacing_ratio, Config.box_names, scale, design_folder)
        store_design(self, "Back", num_deps, num_cols, spacing_ratio, Config.box_names, scale, design_folder)
        store_design(self, "Left", num_deps, num_rows, spacing_ratio, Config.box_names, scale, design_folder)
        store_design(self, "Right", num_deps, num_rows, spacing_ratio, Config.box_names, scale, design_folder)
        return

###################
# Package Library #
###################
def store_design(package, design_name, num_r, num_c, spacing_ratio, box_names, scale, design_folder):
    # Adjust scale
    box_size = 13.0 * scale
    label_size = 1.75 * scale
    stroke = 0.125 * scale
    spacing = 1.25 * scale
    x_spacing = spacing * spacing_ratio
    y_spacing = spacing

    # Extend box names
    box_names = Config.box_names + Config.box_names

    # Store RxC design
    name = f"{package.name}_{design_name}"
    random.shuffle(box_names)
    num_boxes = num_r * num_c
    layout = Layout.Layout(name, num_r, num_c, box_names[:num_boxes], box_size, stroke, x_spacing, y_spacing, "#000000", "#FFFFFF", label_size, _with_labels=True, _with_arrows = False)
    svg = SVG.SVG(name, None, package.external["width"], package.external["height"], f"0 0 {package.external["width"]} {package.external["height"]}", layout.boxes, _with_profile=False, _with_title=False, _with_labels=True)
    design_path = f"{design_folder}/{name}.svg"
    svg.draw(design_path)

    # Store CxR design
    name = f"{package.name}_{design_name}_T"
    random.shuffle(box_names)
    num_boxes = num_c * num_r
    layout = Layout.Layout(name, num_c, num_r, box_names[:num_boxes], box_size, stroke, x_spacing, y_spacing, "#000000", "#FFFFFF", label_size, _with_labels=True, _with_arrows = False)
    svg = SVG.SVG(name, None, package.external["width"], package.external["height"], f"0 0 {package.external["width"]} {package.external["height"]}", layout.boxes, _with_profile=False, _with_title=False, _with_labels=True)
    design_path = f"{design_folder}/{name}.svg"
    svg.draw(design_path)

    return

#FIN