#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Present all images (in order) from the specified LBB box
"""
import os
import cv2
import numpy as np

# Get LBB root
LBBROOT = os.environ.get('LBBROOT')

# Specify LBB boxes folder
LBBBOXES = LBBROOT + '/repo/boxes'

# Specify box (should be from CLI args)
box = 'electrons'
LBBBOX = LBBBOXES + '/' + box 

# Specify README path
readme_path = LBBBOX + '/README.md'

# Read README.md
with open(readme_path) as f:
    readme = f.readlines()

# FIX (alpha in png?)

# Find images in this box (in order)
image_paths = []
for line in readme:
    if(line[:9] == '<img src='):
        fields = line.split('"')
        image_paths.append(LBBBOX + '/' + fields[1])

# Load first image
raw = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)

# Apply alpha (replace 0-alpha with background colour: white)
B, G, R, A = cv2.split(raw)
alpha = A / 255
R = (255 * (1 - alpha) + R * alpha).astype(np.uint8)
G = (255 * (1 - alpha) + G * alpha).astype(np.uint8)
B = (255 * (1 - alpha) + B * alpha).astype(np.uint8)
image = cv2.merge((B, G, R))

# Open full screen display (window)
cv2.namedWindow(box, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(box,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.imshow(box, image)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# Present lecture...

#FIN