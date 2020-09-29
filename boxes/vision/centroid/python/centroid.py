import numpy as np
import cv2

# Get video capture object for camera 0
cap = cv2.VideoCapture(0)

# Create named window for display
cv2.namedWindow('preview')

# Loop until 'q' pressed
while(True):
    # Read most recent frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Computer Vision Code
    # --------------------
    thresh = 128
    ret, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

    # Extract indices (rows, cols) of foreground pixels
    foreground = np.where(binary)
    rows = foreground[0]
    cols = foreground[1]

    # Average row and column indices to get centroid
    cX = np.int(np.mean(cols))
    cY = np.int(np.mean(rows))
    print([cX, cY])

    # Draw centroid on the original RGB image
    frame = cv2.circle(frame, (cX, cY), 7, (0,0,255), thickness=1, lineType=8, shift=0)

    # --------------------

    # Display the resulting frame
    cv2.imshow('preview', binary)

    # Wait for a keypress, and quit if 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()

# Destroy display window
cv2.destroyAllWindows()