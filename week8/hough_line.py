import cv2
import numpy as np
img = cv2.imread('jail.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)

# Canny operator
edges = cv2.Canny(img, 50, 200)

# array of r and theta values
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)

# Probabilistic Hough Line Transform
# lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 50, 10)

for r_theta in lines:
	arr = np.array(r_theta[0], dtype=np.float64)
	r, theta = arr
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*r	# rcos(theta)
	y0 = b*r	# rsin(theta)

	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))

	# draws a red line
	cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

cv2.imwrite('Hough.jpg', img)

