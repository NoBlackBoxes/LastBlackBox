import matplotlib.pyplot as plt
import numpy as np
import cv2

# Get user name
import os
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
box_path = repo_path + '/boxes/python/_data'
image_path = box_path + '/image.jpeg'

#
# Test 1
#

# Load image
image = cv2.imread(image_path)

# Display image (using matplotlib)
plt.imshow(image)
plt.show()

# Challenge: Fix the colors! (and always remeber the insanity of opencv!)

#
# Test 2
#

# Print image shape
print(image.shape)

# Note: which dimension is width? which dimension is height?

# Average each color channel
color_average = np.mean(image, axis=2)

# Average each image column
column_average = np.mean(color_average, axis=1)

# Plot intensity profile using matplotlib
plt.plot(column_average)
plt.show()


#
# Test 3
#

# Open a camera capture session
device_index = 0
capture = cv2.VideoCapture(device_index)

# Open a window
cv2.namedWindow('live')

# Capture images (until the 'q' button is pressed)
while(True):

    # Read most recent frame
    valid, frame = capture.read()

    # Something wrong?
    if (not valid):
        break

    # Display using OpenCV
    cv2.imshow('live', frame)

    # Wait 1 millisecond for a keypress, and quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the caputre
capture.release()

# Destroy display window
cv2.destroyAllWindows()

# Challenge: Display a "running average" of the incoming camera images

#FIN