# Acquire and process ("grayscale threshold") the still image taken from the camera
import time, cv2
import LBB.config as Config
import matplotlib.pyplot as plt

# Specify paths
project_path = f"{Config.repo_path}/boxes/vision/image_processing/python/offline/"

# Open still image and split color channels
bgr = cv2.imread(f"{project_path}/my_00_still.jpg")
B, G, R = cv2.split(bgr)

# Convert to HSV color space and split channels
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)

# Plot RGB vs HSV comparison
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("RGB and HSV Channel Visualization", fontsize=16)

# ---------------- RGB row ----------------

# Original
axes[0, 0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("RGB Image")
axes[0, 0].axis("off")

# R Channel (grayscale)
axes[0, 1].imshow(R, cmap='gray')
axes[0, 1].set_title("R Channel")
axes[0, 1].axis("off")

# G Channel (grayscale)
axes[0, 2].imshow(G, cmap='gray')
axes[0, 2].set_title("G Channel")
axes[0, 2].axis("off")

# B Channel (grayscale)
axes[0, 3].imshow(B, cmap='gray')
axes[0, 3].set_title("B Channel")
axes[0, 3].axis("off")

# ---------------- HSV row ----------------

# Original HSV converted to RGB for correct display
axes[1, 0].imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
axes[1, 0].set_title("HSV Image")
axes[1, 0].axis("off")

# H Channel
axes[1,1].imshow(H, cmap='hsv', vmin=0, vmax=179)
axes[1,1].set_title("H (cyclic)")
axes[1,1].axis("off")

# S Channel
axes[1, 2].imshow(S, cmap='gray')
axes[1, 2].set_title("S Channel")
axes[1, 2].axis("off")

# V Channel
axes[1, 3].imshow(V, cmap='gray')
axes[1, 3].set_title("V Channel")
axes[1, 3].axis("off")

# Save figure
plt.tight_layout()
plt.savefig(f"{project_path}/my_03_rgb_vs_hsv.jpg")

#FIN