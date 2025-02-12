# -*- coding: utf-8 -*-
"""
LBB : Design : Logo Class

@author: kampff
"""

# Import libraries

# Import modules
import LBB.Design.box as Box

# Store logo box parameters
default_box_size = 13.0
default_box_stroke = 0.5
LBB_box_parameters = [[31.111,50.084,"#000000","#FFFFFF"],
                      [44.056,52.834,"#000000","#FFFFFF"],
                      [45.787,43.810,"#000000","#FFFFFF"],
                      [57.432,49.147,"#000000","#FFFFFF"],
                      [52.633,41.232,"#000000","#FFFFFF"],
                      [61.730,44.465,"#000000","#FFFFFF"],
                      [69.113,52.489,"#000000","#FFFFFF"],
                      [72.266,41.664,"#000000","#FFFFFF"],
                      [68.657,34.046,"#000000","#FFFFFF"],
                      [63.446,29.572,"#000000","#FFFFFF"],
                      [58.531,25.627,"#000000","#FFFFFF"],
                      [49.949,23.757,"#000000","#FFFFFF"],
                      [55.655,36.185,"#000000","#FFFFFF"],
                      [45.271,32.576,"#000000","#FFFFFF"],
                      [40.460,22.821,"#000000","#FFFFFF"],
                      [30.438,24.826,"#000000","#FFFFFF"],
                      [35.347,34.227,"#000000","#FFFFFF"],
                      [29.536,38.456,"#000000","#FFFFFF"],
                      [22.519,27.498,"#000000","#FFFFFF"],
                      [16.807,32.443,"#000000","#FFFFFF"],
                      [13.733,37.922,"#000000","#FFFFFF"],
                      [16.807,44.470,"#000000","#FFFFFF"],
                      [49.552,59.843,"#000000","#FFFFFF"],
                      [65.787,57.432,"#000000","#FFFFFF"],
                      [56.365,62.345,"#000000","#FFFFFF"],
                      [61.080,68.178,"#000000","#FFFFFF"],
                      [23.050,47.314,"#000000","#FFFFFF"],
                      [32.787,43.810,"#000000","#FFFFFF"]]

NBB_box_parameters = [[31.111,50.084,"#FFFFFF","#000000"],
                      [44.056,52.834,"#FFFFFF","#000000"],
                      [45.787,43.810,"#FFFFFF","#000000"],
                      [57.432,49.147,"#FFFFFF","#000000"],
                      [52.633,41.232,"#FFFFFF","#000000"],
                      [61.730,44.465,"#000000","#000000"],
                      [69.113,52.489,"#FFFFFF","#000000"],
                      [72.266,41.664,"#FFFFFF","#000000"],
                      [68.657,34.046,"#FFFFFF","#000000"],
                      [63.446,29.572,"#FFFFFF","#000000"],
                      [58.531,25.627,"#FFFFFF","#000000"],
                      [49.949,23.757,"#FFFFFF","#000000"],
                      [55.655,36.185,"#FFFFFF","#000000"],
                      [45.271,32.576,"#000000","#000000"],
                      [40.460,22.821,"#FFFFFF","#000000"],
                      [30.438,24.826,"#FFFFFF","#000000"],
                      [35.347,34.227,"#FFFFFF","#000000"],
                      [29.536,38.456,"#FFFFFF","#000000"],
                      [22.519,27.498,"#FFFFFF","#000000"],
                      [16.807,32.443,"#000000","#000000"],
                      [13.733,37.922,"#FFFFFF","#000000"],
                      [16.807,44.470,"#FFFFFF","#000000"],
                      [49.552,59.843,"#FFFFFF","#000000"],
                      [65.787,57.432,"#FFFFFF","#000000"],
                      [56.365,62.345,"#000000","#000000"],
                      [61.080,68.178,"#FFFFFF","#000000"],
                      [23.050,47.314,"#FFFFFF","#000000"],
                      [32.787,43.810,"#000000","#000000"]]

# Logo Class
class Logo:
    def __init__(self, _name, _x_offset, _y_offset, _box_names, _box_params, _box_size, _box_stroke, _with_labels=False):
        self.name = _name                   # Logo name
        self.x_offset = _x_offset           # Logo offset (x)
        self.y_offset = _y_offset           # Logo offset (y)
        self.box_names = _box_names         # Box names
        self.box_params = _box_params       # Box parameters
        self.box_size = _box_size           # Box size (mm)
        self.box_stroke = _box_stroke       # Box stroke
        self.with_labels = _with_labels     # With box labels?
        self.boxes = self.generate_boxes()  # Logo boxes
        return
    
    def generate_boxes(self):
        boxes = []
        num_boxes = len(self.box_names)
        for i in range(num_boxes):
            name = self.box_names[i]
            x = self.box_params[i][0]
            y = self.box_params[i][1]
            fill = self.box_params[i][2]
            border = self.box_params[i][3]
            if self.with_labels:
                label = name
            else:
                label = None
            box = Box.Box(name, label, 0.0, 0, x + self.x_offset, y + self.y_offset, self.box_size, self.box_size, self.box_stroke, fill, border)
            boxes.append(box)
        return boxes

#FIN