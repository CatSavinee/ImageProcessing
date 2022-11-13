import cv2
import numpy as np
import matplotlib.pyplot as plt

# Open the image
path = r"moon.jpg"
ori_img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(float)

# Edge Sharpening
h, w = img.shape

# define Gaussian filter
smooth = np.array([[1, 2, 1], 
                   [2, 4, 2], 
                   [1, 2, 1]])  
smooth = smooth / 16

# define laplace filter
second_deriv = np.array([[-1, -1, -1], 
                         [-1, 8, -1], 
                         [-1, -1, -1]])  

# define new images
smooth_img = np.zeros((h, w))
sharpen_ed = np.zeros((h, w))

# starting at index 1
for i in range(1, h-1):
    for j in range(1, w-1):
        conv = (np.sum(np.multiply(smooth, img[i-1:i+2, j-1:j+2])))
        smooth_img[i,j] = conv

for i in range(1, h-1):
    for j in range(1, w-1):
        conv = (np.sum(np.multiply(second_deriv, smooth_img[i-1:i+2, j-1:j+2])))
        sharpen_ed[i,j] = conv

weight = 5
sharpen_img = img - (weight*sharpen_ed)
cv2.imwrite('edge_sharp.jpg', sharpen_img)


