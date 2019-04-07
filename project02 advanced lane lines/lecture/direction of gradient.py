import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import pdb;


# Read in an image and grayscale it
# image = cv2.imread('signs_vehicles_xygrad.png')  # BGR
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # cv2.COLOR_BGR2GRAY

image = mpimg.imread('signs_vehicles_xygrad.png')
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold2(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    binary_output = np.copy(img) # Remove this line
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = img

    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # pdb.set_trace()
    # Return the binary image
    return binary_output


# Run the function
dir_binary = dir_threshold(image, sobel_kernel=3, thresh=(1.2, 1.8))  # 1.5707 = 90 degree
dir_binary = dir_threshold(image, sobel_kernel=3, thresh=(-0.2, 0.2))  # 1.5707 = 90 degree

dir_binary = dir_threshold(image, sobel_kernel=3, thresh=(0.7, 1.3))  # 1.5707 = 90 degree

# Plot the result
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(dir_binary, cmap='gray')
ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

plt.show()

print('--------------------')
# pdb.set_trace()
print('done')
print('--------------------')
