# Generate "Build a brain" Layout

# Import Libraries
import csv

# Import Modules
import LBB.config as Config

# Specify paths
resources_path = f"{Config.course_root}/_resources/designs/layout"
box_parameters_path = resources_path + "/box_parameters_buildabrain.csv"

# List all "boxes" in order of opening
boxes = [
    'Sensors',
    'Motors',
    'Transistors',
    'Computers',
    'Robotics',
    'Systems',
    'Networks',
    'Intelligence'
]
num_boxes = len(boxes)
print(num_boxes)

# Defaults
box_size = 13.0
box_stroke = 0.125
box_spacing = 1.25
num_rows = 2
num_cols = 4

# Generate box parameters
box_parameters = []
x = 0.0
y = 0.0
x_step = box_size + box_spacing
y_step = box_size + box_spacing
x_offset = box_stroke
y_offset = box_stroke
for i in range(num_boxes):

    # Determine arrow state: 0: none, 1: right, -1: left, 2: down 
    if (i % num_cols) == (num_cols - 1):  # Last col
        arrow_state = 0
    else:
        arrow_state = 1

    # Write box parameters
    name = boxes[i]
    box_parameters.append([name, x + x_offset, y + y_offset, box_size, box_size, box_stroke, '#000000', '#FFFFFF', arrow_state, 1.75])

    # Set next X,Y (and steps)
    if (i % num_cols) == (num_cols - 1): # Last col
        x = 0.0
        y = y + y_step
    else:
        x = x + x_step

# Save box parameters
with open(box_parameters_path, 'w') as f:
    csv.writer(f).writerows(box_parameters)

#FIN