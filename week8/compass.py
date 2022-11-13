import cv2
import numpy as np
import matplotlib.pyplot as plt

# Open the image
path = r"sunflower.jpg"
ori_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(float) 

# Compass Operator
h, w = img.shape
# define filters
h0 = np.array([[-1, 0, 1], 
               [-2, 0, 2], 
               [-1, 0, 1]])  

h1 = np.array([[-2, -1, 0], 
               [-1, 0, 1], 
               [0, 1, 2]])  

h2 = np.array([[-1, -2, -1], 
               [0, 0, 0], 
               [1, 2, 1]])

h3 = np.array([[0, -1, -2], 
               [1, 0, -1], 
               [2, 1, 0]]) 

mask = [h0, -h0, h1, -h1, h2, -h2, h3, -h3] 

# define the variable for each mask in the convolution
h0_conv = 0 
h1_conv = 0
h2_conv = 0 
h3_conv = 0 
h4_conv = 0 
h5_conv = 0
h6_conv = 0
h7_conv = 0
conv = [h0_conv, h1_conv, h2_conv, h3_conv, h4_conv, h5_conv, h6_conv, h7_conv]

# define empty matrix
h0_img = np.zeros((h, w))
h1_img = np.zeros((h, w))
h2_img = np.zeros((h, w))
h3_img = np.zeros((h, w))
h4_img = np.zeros((h, w))
h5_img = np.zeros((h, w))
h6_img = np.zeros((h, w))
h7_img = np.zeros((h, w))
emp = [h0_img, h1_img, h2_img, h3_img, h4_img, h5_img, h6_img, h7_img]

gradient_img = np.zeros((h, w))

# starting at index 1
for i in range(1, h-1):
    for j in range(1, w-1):
        for k in range(8):
            conv[k] = abs(np.sum(np.multiply(mask[k], img[i-1:i+2, j-1:j+2])))
            emp[k][i-1,j-1] = conv[k]

        # Edge Magnitude
        Es = max(conv[0],conv[1],conv[2],conv[3],conv[4],conv[5],conv[6],conv[7])
        gradient_img[i-1,j-1] = Es

plt.subplot(121)
plt.title("original image")
plt.imshow(ori_img, cmap='gray')

plt.subplot(122)
plt.title("compass operator")
plt.imshow(gradient_img, cmap='gray')

plt.show()

