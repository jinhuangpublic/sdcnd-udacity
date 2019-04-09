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

from utils import show_two_images


from matplotlib.pyplot import  figure



logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)





def calibrate_camera(reuse=True):
    pickle_filename = 'calibration_pickle.p'
    if reuse and os.path.isfile(pickle_filename):
        dist_pickle = pickle.load(open(pickle_filename, "rb"))

        ret = dist_pickle["ret"]
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
        rvecs = dist_pickle["rvecs"]
        tvecs = dist_pickle["tvecs"]
        return ret, mtx, dist, rvecs, tvecs

    x = 9
    y = 6
    objp = np.zeros((x*y,3), np.float32)
    objp[:,:2] = np.mgrid[0:x, 0:y].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/cal*.jpg')

    image = mpimg.imread(images[0])
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # cv2.COLOR_BGR2GRAY
    img_size = (image.shape[1], image.shape[0])

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (x,y), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # # Draw and display the corners
            # cv2.drawChessboardCorners(img, (8,6), corners, ret)
            # #write_name = 'corners_found'+str(idx)+'.jpg'
            # #cv2.imwrite(write_name, img)
            # cv2.imshow('img', img)
            # plt.show()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dist_pickle = {
        'ret': ret,
        'mtx': mtx,
        'dist': dist,
        'rvecs': rvecs,
        'tvecs': tvecs
    }
    pickle.dump(dist_pickle, open("calibration_pickle.p", "wb"))
    return ret, mtx, dist, rvecs, tvecs


def undistort(image, mtx, dist, debug=False):
    frame = cv2.undistort(image, mtx, dist, newCameraMatrix=mtx)

    if debug:
        show_two_images(image, frame, "original vs. calibrated")

    return frame


def warp_to_birdeye(image, debug=False):
    h, w = image.shape[0], image.shape[1]

    # # calibrated for image1
    # src = np.array([
    #     [310, 650],
    #     [1001, 650],
    #     [565, 470],
    #     [719, 470],
    # ], np.float32)

    # calibrated for image2
    src = np.array([
        [313, 650],
        [1001, 650],
        [565, 470],
        [722, 470],
    ], np.float32)

    width = w * 0.15
    height = h * 0.05
    dst = np.array([
        [0 + width, h - height],
        [w - width, h - height],
        [0 + width, 0 + height],
        [w - width, 0 + height]
    ], np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)

    if debug:
        f, axarray = plt.subplots(1, 2)
        f.set_facecolor('white')
        axarray[0].set_title('Before perspective transform')
        axarray[0].imshow(image, cmap='gray')
        for point in src:
            axarray[0].plot(*point, '.')
        axarray[1].set_title('After perspective transform')
        axarray[1].imshow(warped, cmap='gray')
        for point in dst:
            axarray[1].plot(*point, '.')
        for axis in axarray:
            axis.set_axis_off()
        plt.show()

    return warped, M, Minv


if __name__ == '__main__':
    # image = mpimg.imread("test_images/test3.jpg")

    image = mpimg.imread("test_images/straight_lines2.jpg")
    image = mpimg.imread("test_images/straight_lines1.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # calibrate camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera()

    calibrated_image = undistort(image, mtx, dist, debug=True)

    # warp
    warp = warp_to_birdeye(calibrated_image, debug=True)

    print(image.shape)
    print(warp.shape)

    print("abc")
    # pdb.set_trace()
    # undistort(image, mtx, dist, debug=True)

