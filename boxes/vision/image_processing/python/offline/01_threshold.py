# Acquire and process ("grayscale threshold") the still image taken from the camera
import cv2
import LBB.config as Config

# Specify paths
project_path = f"{Config.repo_path}/boxes/vision/image_processing/python/offline/"

# Set threshold level
threshold_level = 127

# Open still image
frame = cv2.imread(f"{project_path}/my_00_still.jpg")

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Apply binary threshold
_, binary = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_BINARY)

# Save processed image
cv2.imwrite(f"{project_path}/my_01_threshold.jpg", binary)

#FIN