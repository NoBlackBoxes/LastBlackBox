# Acquire and process ("Canny Edge Detection") the still image taken from the camera
import time, cv2
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

# Find edges
edges = cv2.Canny(binary, 50, 150)

# Save processed image
cv2.imwrite(f"{project_path}/my_02_edges.jpg", edges)

#FIN