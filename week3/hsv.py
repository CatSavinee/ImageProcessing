import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread('images.jpg')
cv.imshow('Original',img)

# convert RGB to HSV by openCV
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow("HSV", hsv_img)

# convert RGB to HSV by formula
def hsv_conversion(r, g, b):
    r, g, b = r/255, g/255, b/255
    mx = np.max([r, g, b])
    mn = np.min([r, g, b])
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*255
    v = mx*255

    return [h/2, s, v]

h = np.zeros(img.shape[:2])
s = np.zeros(img.shape[:2])
v = np.zeros(img.shape[:2])

row, col = img.shape[0], img.shape[1]
for i in range (0,row):
    h1 = []
    s1 = []
    v1 = []
    for j in range (0,col):
        r = img[i,j][2]
        g = img[i,j][1]
        b = img[i,j][0]
        hsv = hsv_conversion(r,g,b)
        h1.append(hsv[0])
        s1.append(hsv[1])
        v1.append(hsv[2])
    h[i] = h1
    s[i] = s1
    v[i] = v1

hsv = np.round(cv.merge((v,s,h))).astype(np.int)
plt.imshow(hsv)
plt.title("HSV conversion by formula")
plt.show()


