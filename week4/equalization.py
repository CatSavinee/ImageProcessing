import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('boy.png',0)
cv.imshow("Original",img)

# pixels in 1D array 
flat = img.flatten()
plt.hist(flat, bins=50,color="g")

def histo(image_array, bins):
    histo = np.zeros(bins)
    for i in image_array:
        histo[i] += 1
    return histo

hist = histo(flat, 256)

# cumulative function
def cumu(var):
    cumvar = np.zeros(var.shape)
    cumvar[0] = var[0]
    for i in range(1,256):
        cumvar[i] = cumvar[i-1] + var[i]
    return np.array(cumvar)

cumu_sum = cumu(hist)
plt.plot(cumu_sum,color="b")

# normalization to the image
data = (cumu_sum - cumu_sum.min()) * 255
n = cumu_sum.max() - cumu_sum.min()
cumu_sum = (data / n).astype('uint8')
plt.plot(cumu_sum,"r")

# apply cumulative to each index
new_img = cumu_sum[flat]
new_img = np.reshape(new_img, img.shape)


plt.subplot(121)
plt.imshow(img, cmap='gray')

plt.subplot(122)
plt.imshow(new_img, cmap='gray')

plt.show()