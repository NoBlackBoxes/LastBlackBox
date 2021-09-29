import curses
import numpy as np
import cv2

# Get the curses screen (terminal window)
screen = curses.initscr()

# Get the screen size (in characters)
height, width = screen.getmaxyx()

# Create fullscreen (terminal window)
window = curses.newwin(height, width, 0, 0)

# Get video capture object for camera 0
cap = cv2.VideoCapture(0)

# Loop until 'q' pressed
while(True):
    # Read most recent frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize to screen size
    resized = cv2.resize(gray, (width, height))

    # Draw frame in terminal window
    for y in range(0, height-1):
        for x in range(0, width-1):
            if(resized[y, x] > 128):
                window.addch(y, x, '8')
            else:
                window.addch(y, x, '.')

    # Refresh
    window.refresh()

# Release the caputre
cap.release()

# Release curses screen
curses.endwin()
