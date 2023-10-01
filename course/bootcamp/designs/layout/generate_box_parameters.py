# Generate LastBlackBox Layout
import os
import csv
import numpy as np

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
boxes_path = repo_path + '/boxes'
layout_path = repo_path + '/course/bootcamp/designs/layout'
box_parameters_path = layout_path + "/box_parameters.csv"
svg_path = layout_path + "/output.svg"

# List all "boxes" in order of opening
boxes = [
    'Atoms',
    'Electrons',
    'Magnets',
    'Sensors',
    'Motors',
    'Transistors',
    'Computers',
    'Behaviour',
    'Power',
    'Systems',
    'Linux',
    'Python',
    'Networks',
    'Websites',
    'Servers',
    'Audio',
    'Vision',
    'Intelligence'
]
num_boxes = len(boxes)

# Defaults
box_size = 13.0
box_stroke = 0.125
num_rows = 3
num_cols = 6

# Generate box parameters
box_parameters = []
x = 0.0
y = 0.0
x_step = box_size + 1.25
y_step = box_size + 1.25
x_offset = box_stroke
y_offset = box_stroke
for i in range(num_boxes):

    # Determine arrow state: 1: right, -1: left, 0: down, 2: none 
    if (i == num_rows*num_cols-1):           # Last box
        arrow_state = 2
    elif (i % num_cols) == (num_cols - 1):  # Last col
        arrow_state = 0
    else:
        arrow_state = np.sign(x_step)

    # Write box parameters
    name = boxes[i]
    box_parameters.append([name, x + x_offset, y + y_offset, '000000', arrow_state])

    # Set next X,Y (and steps)
    if (i % num_cols) == (num_cols - 1): # Last col
        x_step = -1 * x_step
        y = y + y_step
    else:
        x = x + x_step

# Save box parameters
with open(box_parameters_path, 'w') as f:
    csv.writer(f).writerows(box_parameters)

#FIN