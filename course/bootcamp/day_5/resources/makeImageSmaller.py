# make image smaller
import cv2

im = cv2.imread("test.jpg")
im = cv2.resize(im, (640, 480))
cv2.imwrite("test_small.jpg", im)