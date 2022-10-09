import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread('images.jpg')
cv.imshow('Original',img)

# convert BGR to HSI by formula
b = img[:,:,0] / 255.0
g = img[:,:,1] / 255.0
r = img[:,:,2] / 255.0

k = 1-np.max(img/255, axis=2)
c = (1-r-k)/(1-k)
m = (1-g-k)/(1-k)
y = (1-b-k)/(1-k)

CMYK_image= (np.dstack((c,m,y,k)) * 255).astype(np.uint8)
cv.imshow("CMYK",CMYK_image)
cv.imshow("C channel",c)
cv.imshow("M channel",m)
cv.imshow("Y channel",y)
cv.waitKey(0)


