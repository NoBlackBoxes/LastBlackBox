#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert all images (in order) from the specified LBB box
"""
import os
import sys
import shutil
import cv2
import numpy as np

# Get VK root
VKROOT = os.environ.get('VKROOT')

# Add VK python libraries path
os.sys.path.append(VKROOT + '/repo/resources/python/libraries')

# Import VK python libraries
import img_utils as img

# Get LBB root
LBBROOT = os.environ.get('LBBROOT')

# Specify LBB boxes folder
LBBBOXES = LBBROOT + '/repo/boxes'

# Specify box (should be from CLI args) and "_images" and output "_vk" folders
box = 'electrons'
LBBBOX = LBBBOXES + '/' + box 
_IMAGES = LBBBOX + "/_images"
_VK = _IMAGES + "/_vk"

# Create VK images folder (after removing a possible previous folder)
if os.path.exists(_VK) and os.path.isdir(_VK):
    shutil.rmtree(_VK)
os.mkdir(_VK)

# Specify README path
readme_path = LBBBOX + '/README.md'

# Read README.md
with open(readme_path) as f:
    readme = f.readlines()

# Find images in this box (in order)
image_paths = []
for line in readme:
    if(line[:9] == '<img src='):
        fields = line.split('"')
        image_path = LBBBOX + '/' + fields[1]
        
        # Check if image exists
        if(os.path.isfile(image_path)):
            image_paths.append(LBBBOX + '/' + fields[1])

# Convert to VKIMAGEs
count = 0
for path in image_paths:

    # Get folder path
    folder= os.path.dirname(path)

    # Get image name
    name = os.path.basename(path).split('.')[0]
    
    # Set output path
    output_path = folder + "/_vk/" + str(count) + "_" + name + ".vkimage"
    
    # Rescale

    # Convert
    img.Convert_PNG_to_VK(path, output_path)

    # Increment counter
    count = count + 1

#FIN