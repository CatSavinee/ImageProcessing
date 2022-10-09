import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms
import cv2 as cv

# main image
img = cv.imread("300.png")

# reference image
ref = cv.imread("boy.png")

matched = match_histograms(img, ref, multichannel=True)

plt.subplot(131)
plt.title("Original Image")
plt.imshow(img)

plt.subplot(132)
plt.title("Reference Image")
plt.imshow(ref)

plt.subplot(133)
plt.title("Matched Image")
plt.imshow(matched)

plt.show()

for i, img in enumerate((img, ref, matched)):
  img_hist, bins = exposure.histogram(img[..., i])
  plt.subplot(131)
  plt.plot(bins, img_hist / img_hist.max())
  plt.legend(["original","reference","matched"])
  plt.title("Histogram")
  
  img_cdf, bins = exposure.cumulative_distribution(img[..., i])
  plt.subplot(132)
  plt.plot(bins, img_cdf)
  plt.legend(["original","reference","matched"])
  plt.title("Cumulative graph")
  

plt.show()
