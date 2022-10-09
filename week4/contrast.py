import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('giraffe.jpg',0)
cv.imshow("Original",img)

row = img.shape[0]
col = img.shape[1]
# zeros array stores the stretched image
cont = np.zeros((row,col),dtype = 'uint8')
minimg = np.min(img)
maximg = np.max(img)

# contrast stretching
for i in range(row):
    for j in range(col):
       cont[i,j] = 255*(img[i,j]-minimg)/(maximg-minimg)

# Modified contrast stretching
# percentile at 1% and 99%
mod_cont = np.zeros((row,col),dtype = 'uint8')
minimg = np.percentile(img,1)
maximg = np.percentile(img,99)
for i in range(row):
    for j in range(col):
       mod_cont[i,j] = 255*(img[i,j]-minimg)/(maximg-minimg)


# Displat the stretched image
cv.imshow('Contrast Stretching',cont)
cv.imshow('Modified Contrast Stretching',mod_cont)
cv.waitKey(0)

plt.subplot(221)
plt.hist(img.ravel(),256,[0,256])
plt.title("Histogram of original image")

plt.subplot(222)
plt.hist(cont.ravel(),256,[0,256])
plt.title("Histogram from contrast stretching")

plt.subplot(223)
plt.hist(mod_cont.ravel(),256,[0,256])
plt.title("Histogram from modified contrast stretching")

plt.subplot(224)
plt.hist(img.ravel(),256,[0,256],color="g")
plt.hist(cont.ravel(),256,[0,256],color="black")
plt.hist(mod_cont.ravel(),256,[0,256],color="r")
plt.title("Comparision before and after")
plt.legend(["original",'contrast','modified'])

plt.show()