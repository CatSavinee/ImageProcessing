import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter 

path = r"unknown.jpg"
# input image
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(float) 

# prepare the array for new image from filtering
row = img.shape[0]
col = img.shape[1]
max_fil = np.zeros((row,col))
min_fil = np.zeros((row,col))
med_fil = np.zeros((row,col))
w = np.array([[4,2,4],
            [2,1,2],
            [4,2,4]])   # weight matrix
weight_fil = np.zeros((row,col))

# filter the image by using mask
for i in range(1,row-1):
    for j in range(1,col-1):
        mask = np.array([[[img[i-1,j-1],img[i-1,j],img[i-1,j+1]]],
                        [[img[i,j-1],img[i,j],img[i,j+1]]],
                        [[img[i+1,j-1],img[i+1,j],img[i+1,j+1]]]])

        w = w.ravel()
        mask = mask.ravel()
        emp = []
        for k in range(9):
            emp.extend([mask[k]]*w[k])
            emp.sort()

        # replace the pixels in each filtered images
        mask.sort()
        max_fil[i][j] = max(mask)
        min_fil[i][j] = min(mask)
        med_fil[i][j] = mask[4]
        weight_fil[i][j] = emp[8]

# filter using library
img1 = Image.open(path) 
img_fil = img1.filter(ImageFilter.MedianFilter(size = 3)) 

def plot_images(img1: np.array, img2: np.array, img3: np.array, img4: np.array, img5: np.array):
    _, ax = plt.subplots(2, 3, figsize=(12, 6))
    ax[0][0].imshow(img1, cmap='gray')
    ax[0][0].set_title("original")

    ax[0][1].imshow(img2, cmap='gray')
    ax[0][1].set_title("maximum filter")

    ax[0][2].imshow(img3, cmap='gray')
    ax[0][2].set_title("minimum filter")

    ax[1][0].imshow(img4, cmap='gray')
    ax[1][0].set_title("median filter")

    ax[1][1].imshow(img_fil, cmap='gray')
    ax[1][1].set_title("median filter by lib.")

    ax[1][2].imshow(img5, cmap='gray')
    ax[1][2].set_title("weight median filter")
    plt.show()

plot_images(img1=img,img2=max_fil,img3=min_fil,img4=med_fil,img5=weight_fil)

