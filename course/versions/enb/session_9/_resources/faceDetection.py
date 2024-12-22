# Face detection code with sanity check
import numpy as np 
import cv2 

# Load test image
im = cv2.imread("test.jpg")

det = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
rects = det.detectMultiScale(gray, 
    scaleFactor=1.1, 
    minNeighbors=5, 
    minSize=(200, 200), # adjust to your image size, maybe smaller, maybe larger?
    flags=cv2.CASCADE_SCALE_IMAGE)

for (x, y, w, h) in rects:
    # x: x location
    # y: y location
    # w: width of the rectangle 
    # h: height of the rectangle
    # Remember, order in images: [y, x, channel]
    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 20)

cv2.imwrite("test_face.jpg", im)