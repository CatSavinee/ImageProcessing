import cv2 as cv
import numpy as np

# from matplotlib import pyplot as plt
# img = cv.imread('cell.jfif',0)
# hist,bins = np.histogram(img.flatten(),256,[0,256])
# cdf = hist.cumsum()
# cdf_normalized = cdf * float(hist.max()) / cdf.max()

# cdf_m = np.ma.masked_equal(cdf,0)
# cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
# cdf = np.ma.filled(cdf_m,0).astype('uint8')

# plt.t(cdf)
# plt.hist(img.flatten(),256,[0,256], color = 'r')
# plt.xlim([0,256])
# plt.legend(('cdf','histogram'), loc = 'upper left')
# plt.show()

# ------------------------------
# img = cv.imread('boy.png',0)
# equ = cv.equalizeHist(img)
# equ_img = np.hstack((img,equ)) #stacking images side-by-side
# cv.imwrite('equ.png ',equ_img )
