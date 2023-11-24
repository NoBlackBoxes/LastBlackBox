# Load an image, process it with numpy and scikit-image, and track colored object

import os
import numpy as np
import skimage as ski

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
image_path = repo_path + '/course/bootcamp/day_5/resources/images/NB3_ears_mount.jpg'
output_path = repo_path + '/_tmp/output.png'

# Load RGB image
rgb = ski.io.imread(image_path)

# Convert RGB to HSV
hsv = ski.color.rgb2hsv(rgb)

# Seperate the Hue, Saturation, and Value channels
H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

# Define HSV thresholds
H_min, H_max = 20/360, 40/360
S_min, S_max = 100/255, 1.0
V_min, V_max = 100/255, 1.0

# Binary threshold on Hue, Saturation, and Value images
binary =    (((H >= H_min) & (H <= H_max)) & 
            ((S >= S_min) & (S <= S_max)) & 
            ((V >= V_min) & (V <= V_max))).astype(np.uint8) * 255

# Save the processed image to a file
ski.io.imsave(output_path, binary)

# Find binary regions
label_image = ski.measure.label(binary)
blobs = ski.measure.regionprops(label_image)
num_blobs = len(blobs)

# Find largest binary region
largest_area = 0
largest_id = -1
id = 0
for blob in blobs:
    if blob.area > largest_area:
        largest_id = id
        largest_area = blob.area
    id = id + 1
largest_blob = blobs[largest_id]

# Report
print("Blob Area: {0:.1f}, Blob Centroid: {1:.2f},{2:.2f}".format(largest_blob.area, largest_blob.centroid[0], largest_blob.centroid[1]))

#FIN