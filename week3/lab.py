import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('images.jpg')
cv.imshow('Original',img)

LAB = cv.cvtColor(img,cv.COLOR_BGR2LAB)
cv.imshow('LAB from openCV',LAB)

img = img.astype(np.uint8)

b = img[:,:,0] / 255
g = img[:,:,1] / 255
r = img[:,:,2] / 255
x = ((0.412453*r) + (0.357580*g) + (0.180423*b)) / 0.950455
y = ((0.212671*r) + (0.715160*g) + (0.072169*b)) / 1.0
z = ((0.019334*r) + (0.119193*g) + (0.950227*b)) / 1.088753

if y.any() > 0.008856:
    y1 = np.power(y,1/3)
    L = ((116*y1)-16)*(255/100)
else:
    L = (903.3*y)*(255/100)

def equation(a):
    if a.any() > 0.008856:
        a = np.power(a,1/3)
    else:
        a = (7.787*a) + (16/116)
    return a

A = (500*(equation(x) - equation(y))) + 128
B = (200*(equation(y) - equation(z))) + 128

cv.imshow("L channel",L)
cv.imshow("A channel",A)
cv.imshow("B channel",B)

LAB1 = np.dstack((B,A,L)).astype(np.uint8)
plt.imshow(LAB1)
plt.title("LAB conversion by formula")
plt.show()


