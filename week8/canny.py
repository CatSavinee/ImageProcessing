import cv2
import matplotlib.pyplot as plt

# Open the image
img = cv2.imread('kratong.png', cv2.IMREAD_GRAYSCALE)

# Canny Operator
canny = cv2.Canny(img, 50, 200)

plt.subplot(121)
plt.title('original image')
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.title('canny operator')
plt.imshow(canny, cmap='gray')
plt.show()


