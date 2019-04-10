import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
from collections import deque
import numpy as np
from moviepy.editor import VideoFileClip
from os.path import join, basename

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import pdb

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

import os.path


# # Imports for Jupyter
# %matplotlib inline

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def show_two_images(image1, image2, title="Two Images", cmap1=None, cmap2=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.suptitle(title, fontsize=16)
    f.tight_layout()
    ax1.imshow(image1, cmap=cmap1)
    ax1.set_title('First Image', fontsize=50)
    ax2.imshow(image2, cmap=cmap2)
    ax2.set_title('Second Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def show_image(image, title="Two Images", cmap=None):
    f, ax1 = plt.subplots(1, 1, figsize=(14, 9))
    f.suptitle(title, fontsize=16)
    f.tight_layout()
    ax1.imshow(image, cmap=cmap)
    # ax1.set_title('First Image', fontsize=50)
    # ax2.imshow(image2, cmap=cmap2)
    # ax2.set_title('Second Image', fontsize=50)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()



def add_info(image, image1, image2=None, image3=None, data={}):
    # frame = image.copy()
    h, w = image.shape[0], image.shape[1]
    thumbnail_ratio = 0.2
    thumbnail_h, thumbnail_w = int(thumbnail_ratio * h), int(thumbnail_ratio * w)
    off_x, off_y = 20, 15

    # Add a bar
    mask = image.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumbnail_h+2*off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    frame = cv2.addWeighted(src1=mask, alpha=0.2, src2=image, beta=0.8, gamma=0)

    # thumb1
    thumb1 = cv2.resize(image1, dsize=(thumbnail_w, thumbnail_h))
    frame[off_y:thumbnail_h+off_y, off_x:off_x+thumbnail_w, :] = thumb1

    # thumb2
    if image2 is not None:
        thumb2 = cv2.resize(image2, dsize=(thumbnail_w, thumbnail_h))
        frame[off_y:thumbnail_h+off_y, 2*off_x+thumbnail_w:2*(off_x+thumbnail_w), :] = thumb2

    # thumb3
    if image3 is not None:
        thumb3 = cv2.resize(image3, dsize=(thumbnail_w, thumbnail_h))
        frame[off_y:thumbnail_h+off_y, 3*off_x+2*thumbnail_w:3*(off_x+thumbnail_w), :] = thumb3

    for i, key in enumerate(data):
        value = data[key]
        cv2.putText(img=frame,
                    text=f"{key}: {value}",
                    org=(860, 60 + i * 30),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(255, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA)

    return frame


def binary_to_image(binary_matrix):
    return np.dstack([binary_matrix, binary_matrix, binary_matrix]) * 255


def gray_to_image(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)



if __name__ == '__main__':
    image1 = mpimg.imread("test_images/test6.jpg")
    gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    frame = add_info(image1, image1, image1, image1)

    show_image(frame)

    #
    # image2 = mpimg.imread("test_images/test5.jpg")
    #
    # show_two_images(image1, image2)
    # show_two_images(gray, image2, cmap1='gray')

    # plt.imshow(image1)
    # cv2.waitKey(10 * 1000)
    # plt.show()


