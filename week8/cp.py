import cv2
import numpy as np
import matplotlib.pyplot as plt

# filters
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
      
def convol(img,h,w,mask):
    fil_img = np.zeros((h,w))
    for i in range(1, h-1):
        for j in range(1, w-1):
            fil_conv = abs(np.sum(np.multiply(mask, img[i-1:i+2, j-1:j+2])))
            fil_img[i-1,j-1] = fil_conv
    return fil_conv


# Open the image
path = r"sunflower.jpg"
ori_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(float)
h , w = img.shape

for i in range(8):
    total = []
    total.append(convol(img, h, w, mask[i]))
    
    