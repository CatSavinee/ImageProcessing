import cv2
import numpy as np
import matplotlib.pyplot as plt

# input image
img = cv2.imread('malfoy.jpg', cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
width = int(img.shape[1] / 2)
height = int(img.shape[0] / 2)
dim = (width, height)
img = cv2.resize(img, dim)

# filters
def gaussuian_filter(kernel_size, sigma=1, muu=0):
 
    # Initializing value of x,y as grid of kernel size
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)
 
    # lower normal part of gaussian
    normal = 1/(2.0 * np.pi * sigma**2)
 
    # Calculating Gaussian filter
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal
    return gauss

def box_filter(kernel_size):
    # Create the matrix of ones and average by size
    box = np.ones((kernel_size,kernel_size))*(1/(kernel_size*kernel_size))
    return box

def laplacian_filter(kernel_size):
    # Define each matrix of laplacian filter
    if kernel_size == 3:
        lapla = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]])
    elif kernel_size == 5:
        lapla = np.array([[0, 0, -1, 0, 0],
                          [0, -1, -2, -1, 0],
                          [-1, -2, 16, -2, -1],
                          [0, -1, -2, -1, 0],
                          [0, 0, -1, 0, 0]])
    else:
        print("there is not available kernel")
    return lapla

# Select the type of fiters
def select_kernel(kernel_size, filters):    #filters: 0 = box, 1 = gaussian, 2 = laplacian
    if filters == 0:
        kernel = box_filter(kernel_size)
    elif filters ==1:
        kernel = gaussuian_filter(kernel_size)
    elif filters == 2:
        kernel = laplacian_filter(kernel_size)
    else:
        print("error is found")
    return kernel

# Create filter size due to the image size
def filter_img_size(img_shape, kernel_size: int):
    # Initializing the number of pixels in each axis
    num_pix_y = 0
    num_pix_x = 0

    for i in range(img_shape[0]):
        added = i + kernel_size
        if added <= img_shape[0]:
            num_pix_y += 1

    for j in range(img_shape[1]):
        added = j + kernel_size
        if added <= img_shape[1]:
            num_pix_x += 1

    return (num_pix_y, num_pix_x)

# Convolution kernel matrix and image
def convol(img: np.array, kernel: np.array):
    sqr_size = filter_img_size(
        img_shape=img.shape,
        kernel_size=kernel.shape[0]
    )
    print("image size = %s, kernel size = %s" % (img.shape, kernel.shape))

    k = kernel.shape[0]

    # 2D array of zeros
    convol_img = np.zeros(shape=sqr_size)
    
    # Iterate over the rows
    for i in range(sqr_size[0]):
        # Iterate over the columns
        for j in range(sqr_size[1]):
            # Get the current matrix
            mat = img[i:i+k, j:j+k]
            
            # Apply the convolution -multiplication and summation of the result
            convol_img[i, j] = np.sum(np.multiply(mat, kernel))

    return convol_img

# Convolution by using tool from OpenCV
def convol_cv(image,kernel):
    filtered = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
    return filtered

# Plot image for comparison
def plot_images(img1: np.array, img2: np.array):
    _, ax = plt.subplots(1, 3, figsize=(12, 6))
    ax[0].imshow(img1, cmap='gray')
    ax[0].set_title("original")

    ax[1].imshow(img2, cmap='gray')
    ax[1].set_title("guassian filter 3*3")

    plt.show()

# main area to use the functions
filter_img = convol(img=np.array(img), kernel=select_kernel(3,2))
plot_images(img, filter_img)
#plot_images(img, convol_cv(img,select_kernel(5,1)))