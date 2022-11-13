import cv2
import numpy as np
import matplotlib.pyplot as plt

# Open the image
path = r"coins.png"
ori_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(float) 

# Robert Operator
h, w = img.shape
# define filters
horizontal_cross = np.array([[0, 0, 0], 
                             [0, 0, 1], 
                             [0, -1,0]])  

vertical_cross = np.array([[0, 0, 0], 
                           [0, 1, 0], 
                           [0, 0, -1]])  

# define new images
horizontal_img = np.zeros((h, w))
vertical_img = np.zeros((h, w))
gradient_img = np.zeros((h, w))

# starting at index 1
for i in range(1, h-1):
    for j in range(1, w-1):
        horizontal_conv = abs(np.sum(np.multiply(horizontal_cross, img[i-1:i+2, j-1:j+2])))
        vertical_conv = abs(np.sum(np.multiply(vertical_cross, img[i-1:i+2, j-1:j+2])))
        
        horizontal_img[i-1,j-1] = horizontal_conv
        vertical_img[i-1,j-1] = vertical_conv

        # Edge Magnitude
        mag = np.sqrt(pow(horizontal_conv, 2.0) + pow(vertical_conv, 2.0))
        gradient_img[i-1,j-1] = mag

plt.subplot(141)
plt.title("Original Image")
plt.imshow(ori_img)

plt.subplot(142)
plt.title("Robert Edge Detection")
plt.imshow(gradient_img, cmap='gray')

plt.subplot(143)
plt.title("horizontal cross")
plt.imshow(horizontal_img, cmap='gray')

plt.subplot(144)
plt.title("vertical cross")
plt.imshow(vertical_img, cmap='gray')

plt.show()