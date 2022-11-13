import cv2
import numpy as np
import matplotlib.pyplot as plt

# Open the image
path = r"ghost.jpg"
ori_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(float) 

# Sobel Operator
h, w = img.shape
# define filters
horizontal = np.array([[-1, 0, 1], 
                       [-2, 0, 2], 
                       [-1, 0, 1]])  

vertical = np.array([[-1, -2, -1], 
                     [0, 0, 0], 
                     [1, 2, 1]])  

# define new images
horizontal_img = np.zeros((h, w))
vertical_img = np.zeros((h, w))
gradient_img = np.zeros((h, w))

# starting at index 1
for i in range(1, h-1):
    for j in range(1, w-1):
        horizontalGrad = (horizontal[0, 0] * img[i - 1, j - 1]) + \
                         (horizontal[0, 1] * img[i - 1, j]) + \
                         (horizontal[0, 2] * img[i - 1, j + 1]) + \
                         (horizontal[1, 0] * img[i, j - 1]) + \
                         (horizontal[1, 1] * img[i, j]) + \
                         (horizontal[1, 2] * img[i, j + 1]) + \
                         (horizontal[2, 0] * img[i + 1, j - 1]) + \
                         (horizontal[2, 1] * img[i + 1, j]) + \
                         (horizontal[2, 2] * img[i + 1, j + 1])
        # horizontal_img[i - 1, j - 1] = abs(horizontalGrad)
        horizontal_img[i-1,j-1] = np.sum(np.multiply(horizontal, img[i-1:i+2, j-1:j+2]))

        verticalGrad = (vertical[0, 0] * img[i - 1, j - 1]) + \
                       (vertical[0, 1] * img[i - 1, j]) + \
                       (vertical[0, 2] * img[i - 1, j + 1]) + \
                       (vertical[1, 0] * img[i, j - 1]) + \
                       (vertical[1, 1] * img[i, j]) + \
                       (vertical[1, 2] * img[i, j + 1]) + \
                       (vertical[2, 0] * img[i + 1, j - 1]) + \
                       (vertical[2, 1] * img[i + 1, j]) + \
                       (vertical[2, 2] * img[i + 1, j + 1])

        # vertical_img[i - 1, j - 1] = abs(verticalGrad)
        vertical_img[i-1,j-1] = np.sum(np.multiply(vertical, img[i-1:i+2, j-1:j+2]))

        # Edge Magnitude
        mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
        gradient_img[i - 1, j - 1] = mag

plt.subplot(121)
plt.title("Original Image")
plt.imshow(ori_img)

plt.subplot(122)
plt.title("Sobel Edge Detection")
plt.imshow(gradient_img, cmap='gray')
plt.show()