import cv2 as cv
img = cv.imread('images.jpg')
cv.imshow('A LONELY MAN',img)

print(img.shape[:])

# separate color channel
blue = img[:,:,0]
green = img[:,:,1]
red = img[:,:,2]
cv.imshow("Blue ch.", blue)
cv.imshow("Green ch.", green)
cv.imshow("Red ch.", red)

# split image
b, g, r = cv.split(img)
cv.imshow("Blue_split", b)
cv.imshow("Green_split", g)
cv.imshow("Red_split", r)
# merge channel
image_merge = cv.merge([b, g, r])

cv.imshow("merge", image_merge)

cv.waitKey(0)