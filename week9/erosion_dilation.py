import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('circle.png', 0)
(thresh, img2) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
img2 = img2/255     # normalize to binary 0,1 image

h,w = img.shape

# Erosion
def erosion(img,h,w,kernel):
    s_element = np.ones((kernel,kernel))
    img_erosion= np.zeros((h,w))
    x  = int((kernel-1)/2)

    for i in range(x, h-x):
        for j in range(x, w-x):
            temp = img[i-x:i+x+1, j-x:j+x+1]
            product = temp*s_element
            img_erosion[i-1,j-1]= np.min(product)
    return img_erosion

# Dilation
def dilation(img,h,w,kernel):
    s_element = np.ones((kernel,kernel))
    img_dilation= np.zeros((h,w))
    x  = int((kernel-1)/2)

    for i in range(x, h-x):
        for j in range(x, w-x):
            temp = img[i-x:i+x+1, j-x:j+x+1]
            product = temp*s_element
            img_dilation[i-1,j-1]= np.max(product)
    return img_dilation
        
eroded_img = erosion(img2,h,w,3)
dilated_img = dilation(img2,h,w,5)

plt.subplot(131)
plt.title('original image')
plt.imshow(img,cmap="gray")

plt.subplot(132)
plt.title('eroded image')
plt.imshow(eroded_img,cmap='gray')

plt.subplot(133)
plt.title('dilated image')
plt.imshow(dilated_img,cmap='gray')

plt.show()


