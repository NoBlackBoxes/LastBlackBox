# -*- coding: utf-8 -*-
"""
Design: Layout Class

@author: kampff
"""

# Import libraries

# Import modules
import LBB.Design.box as Box

# Layout Class
class Layout:
    def __init__(self, _name, _num_rows, _num_cols, _box_names, _box_size, _box_stroke, _box_spacing, _box_fill, _box_border, _label_size, _with_labels, _with_arrows):
        self.name = _name                   # Layout name
        self.num_rows = _num_rows           # Layout #rows
        self.num_cols = _num_cols           # Layout #columns
        self.x_offset = _box_stroke         # Layout offset (x)
        self.y_offset = _box_stroke         # Layout offset (y)
        self.box_names = _box_names         # Box names
        self.box_size = _box_size           # Box size (mm)
        self.box_stroke = _box_stroke       # Box stroke
        self.box_spacing = _box_spacing     # Box spacing
        self.box_fill = _box_fill           # Box fill
        self.box_border = _box_border       # Box border
        self.label_size = _label_size       # Label size
        self.with_labels  = _with_labels    # With labels?
        self.with_arrows  = _with_arrows    # With arrows?
        self.boxes = self.generate_boxes()
        return
    
    def generate_boxes(self):
        boxes = []
        x = 0.0
        y = 0.0
        x_step = self.box_size + self.box_spacing
        y_step = self.box_size + self.box_spacing
        num_boxes = len(self.box_names)
        if num_boxes != (self.num_rows * self.num_cols):
            print(f"Invalid number of boxes in layout {self.name}: #boxes = {num_boxes}, num_rows: {self.num_rows}, num_cols: {self.num_cols}")
            exit(-1)
        for i in range(num_boxes):
            name = self.box_names[i]

            # Include arrow?
            if self.with_arrows:
                # Determine arrow state: 0: none, 1: right, -1: left, 2: down
                if (i % self.num_cols) == (self.num_cols - 1):  # Last col
                    arrow_state = 0
                else:
                    arrow_state = 1
            else:
                arrow_state = 0
            
            # Include labels?
            if self.with_labels:
                label = name
            else:
                label = None

            # Create box (and append to list)
            box = Box.Box(name, label, self.label_size, arrow_state, x + self.x_offset, y + self.y_offset, self.box_size, self.box_size, self.box_stroke, self.box_fill, self.box_border)
            boxes.append(box)

            # Set next X,Y (and steps)
            if (i % self.num_cols) == (self.num_cols - 1): # Last col
                x = 0.0
                y = y + y_step
            else:
                x = x + x_step

        # Store boxes
        return boxes
