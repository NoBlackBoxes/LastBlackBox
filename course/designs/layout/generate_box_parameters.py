# Generate LastBlackBox Layout
import csv
import os

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
boxes_path = repo_path + '/boxes'
layout_path = repo_path + '/course/designs/layout'
box_parameters_path = layout_path + "/box_parameters.csv"
svg_path = layout_path + "/output.svg"

# List all "boxes" in order of opening
boxes = [
    'Atoms',
    'Electrons',
    'Magnets',
    'Light',
    'Sensors',
    'Motors',
    'Transistors',
    'Amplifiers',
    'Reflexes',
    'Power',
    'Data',
    'Logic',
    'Memory',
    'FPGAs',
    'Computers',
    'Control',
    'Behaviour',
    'Systems',
    'Linux',
    'Python',
    'Networks',
    'Websites',
    'Servers',
    'Security',
    'Audio',
    'Vision',
    'Learning',
    'Intelligence'
]
num_boxes = len(boxes)

# Defaults
box_size = 13.0
box_stroke = 0.125
num_rows = 4
num_cols = 7

# Generate box parameters
box_parameters = []
for i in range(num_boxes):
    x = box_size * (i%num_cols)
    y = box_size * (i // num_cols)
    x = x + ((i%num_cols) * 1.25)
    y = y + ((i // num_cols) * 1.25)
    x = x + (box_stroke/2.0) + 0.6875
    y = y + (box_stroke/2.0) + 0.6875
    #y = y + (box_stroke/2.0) + 0.6875 + 20.0
    name = boxes[i]
    box_parameters.append([name, x, y, '000000'])

# Save box parameters
with open(box_parameters_path, 'w') as f:
    csv.writer(f).writerows(box_parameters)

#FIN