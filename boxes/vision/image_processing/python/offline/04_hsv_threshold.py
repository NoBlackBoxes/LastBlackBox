# Acquire and process ("grayscale threshold") the still image taken from the camera
import cv2
import numpy as np
import LBB.config as Config

# Specify paths
project_path = f"{Config.repo_path}/boxes/vision/image_processing/python/offline/"

# Open still image
bgr = cv2.imread(f"{project_path}/my_00_still.jpg")

# Convert to HSV (H:0-179, S:0-255, V:0-255)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# Define HSV range for RED color
# - Hue is a circular color space and "red" wraps around zero
# - This requires two thresholds of the image
lower_red_1 = np.array([0, 130, 100])   # Lower bound (H, S, V)
upper_red_1 = np.array([25, 255, 225])  # Upper bound (H, S, V)
lower_red_2 = np.array([160, 130, 100]) # Lower bound (H, S, V)
upper_red_2 = np.array([255, 255, 225]) # Upper bound (H, S, V)

# Threshold for RED
mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
mask = cv2.bitwise_or(mask_1, mask_2)

# Save processed image
cv2.imwrite(f"{project_path}/my_04_hsv_threshold.jpg", mask)

#FIN