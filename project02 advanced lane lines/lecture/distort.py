import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_image2.png') # BGR
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y


def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    image = cv2.undistort(img, mtx, dist, None, mtx)
    #     plt.imshow(dst)

    # 2) Convert to grayscale
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3) Find the chessboard corners
    found_corner, corners = cv2.findChessboardCorners(image, (8,6), None)

    #     import pdb; pdb.set_trace()

    # 4)
    #     cv2.drawChessboardCorners(image, (nx, ny), corners, found_corner)

    g = plt.imshow(image, cmap='gray')


    #     plt.imshow(image)
    plt.plot(corners[0][0][0], corners[0][0][1], '.', markersize=20)
    plt.plot(corners[7][0][0], corners[7][0][1], '.', markersize=20)
    plt.plot(corners[40][0][0], corners[40][0][1], '.', markersize=20)
    plt.plot(corners[47][0][0], corners[47][0][1], '.', markersize=20)

    plt.show()
    # 5)
    #     src_points = np.float32([[,],[,],[,],[,]])
    src_points = np.array([
        corners[0][0],
        corners[7][0],
        corners[40][0],
        corners[47][0]
    ], np.float32)

    #             a) draw corners
    #             b) define 4 source points src = np.float32([[,],[,],[,],[,]])
    #                  Note: you could pick any four of the detected corners
    #                  as long as those four corners define a rectangle
    #                  One especially smart way to do this would be to use four well-chosen
    #                  corners that were automatically detected during the undistortion steps
    #                  We recommend using the automatic detection of corners in your code

    # 6
    dst_points = np.array([
        [250, 250],
        [1000, 250],
        [250, 800],
        [1000, 800],
    ], np.float32)

    # 7
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # 8
    image_size = (image.shape[1], image.shape[0])
    warped = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)

    # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    # 1


    #delete the next two lines
    # M = None
    # warped = np.copy(img)
    # warped = dst
    return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down, cmap='gray')
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

plt.show()
