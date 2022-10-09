import cv2 as cv
import numpy as np

img = cv.imread('images.jpg')
cv.imshow("Original",img)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

threshold = 132
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i,j] > threshold:
            img[i,j] = 255
        else:
            img[i,j] = 0

cv.imshow('Threshold Image',img)

ret, thresh1 = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
cv.imshow('Binary Threshold from openCV', thresh1)

cv.waitKey(0)


