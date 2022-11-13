import cv2
import numpy as np
import matplotlib.pyplot as plt

path = r"cat.jpg"
ori_img = cv2.imread(path)
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(float)

box = np.ones((3,3))*(1/(3*3))

h, w = img.shape
fil_img = np.zeros((h, w))

for i in range(1, h-1):
    for j in range(1, w-1):
        conv = (np.sum(np.multiply(box,img[i-1:i+2, j-1:j+2])))
        fil_img[i,j] = conv

mask = np.subtract(img, fil_img)
weight = 10.0
unsharp_img = (img + (weight*mask))

cv2.imwrite('unsharp.jpg', unsharp_img)



# def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
#     """Return a sharpened version of the image, using an unsharp mask."""
#     # For details on unsharp masking, see:
#     # https://en.wikipedia.org/wiki/Unsharp_masking
#     # https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm
#     blurred = cv2.GaussianBlur(image, kernel_size, sigma)
#     sharpened = float(amount + 1) * image - float(amount) * blurred
#     sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
#     sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
#     sharpened = sharpened.round().astype(np.uint8)
#     if threshold > 0:
#         low_contrast_mask = np.absolute(image - blurred) < threshold
#         np.copyto(sharpened, image, where=low_contrast_mask)
#     return sharpened

# plt.subplot(121)
# plt.imshow(img, cmap='gray')
# plt.subplot(122)
# plt.imshow(unsharp_mask(img,(3,3)), cmap='gray')
# plt.show()
