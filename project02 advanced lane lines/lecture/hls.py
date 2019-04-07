import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import pdb


# def show_two_images(image1, image2, title="Two Images"):
#     f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#     f.suptitle(title, fontsize=16)
#     f.tight_layout()
#     ax1.imshow(image1)
#     ax1.set_title('First Image', fontsize=50)
#     ax2.imshow(image2)
#     ax2.imshow(image2, cmap='gray')
#     ax2.set_title('Second Image', fontsize=50)
#     plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#     plt.show()


def show_two_images(image1, image2, title="Two Images", cmap1=None, cmap2='gray'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.suptitle(title, fontsize=16)
    f.tight_layout()
    ax1.imshow(image1, cmap=cmap1)
    ax1.set_title('First Image', fontsize=50)
    ax2.imshow(image2, cmap=cmap2)
    ax2.set_title('Second Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

    # 3) Return a binary image of threshold result
    return binary_output


def gray_select(img, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    binary_output = np.zeros_like(gray)
    binary_output[(gray > thresh[0]) & (gray <= thresh[1])] = 1

    # 3) Return a binary image of threshold result
    return binary_output


def rgb_select(img, channel=0, thresh=(0, 255)):
    channel = img[:,:,channel]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1

    # 3) Return a binary image of threshold result
    return binary_output


image = mpimg.imread('test6.jpg')  # RGB
# gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # cv2.COLOR_BGR2GRAY
# hls_binary = hls_select(image, thresh=(0, 255))

# hls_threshold = (90, 255)
# hls_binary = hls_select(image, hls_threshold)
# show_two_images(image, hls_binary, f"HLS: {hls_threshold}")
#
# gray_threshold = (180, 255)
# gray_binary = gray_select(image, gray_threshold )
# show_two_images(image, gray_binary, f"GRAY: {gray_threshold}")

for i in range(1):
    rgb_threshold = (200, 255)
    rgb_binary = rgb_select(image, i, rgb_threshold)
    show_two_images(image[:,:,i],
                    rgb_binary,
                    f"RGB | channel={i}, threshold={rgb_threshold}",
                    cmap1=None)


print('--------------------')
# pdb.set_trace()
print('done')
print('--------------------')
