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


import logging

from camera import calibrate_camera, undistort, warp_to_birdeye
from color import compute_color_binary
from utils import show_two_images, add_info, binary_to_image, show_image

from utils import show_two_images

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


ym_per_pix = 30/720  # meters per pixel in y dimension
xm_per_pix = 3.7/700  # meters per pixel in x dimension


class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []

        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None


def find_lane_pixels(binary_warped, debug=False):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        if debug:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
                          (win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
                          (win_xright_high,win_y_high),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        raise Exception("Cannot find left and right lane")

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped, debug=False):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left_fit` and `right_fit` are still none or incorrect
        msg = 'The function failed to fit a line!'
        print('The function failed to fit a line!')
        raise Exception(msg)
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    draw_lane(out_img, left_fit, color=(255, 255, 255), line_width=10)
    draw_lane(out_img, right_fit, color=(255, 255, 255), line_width=10)

    # Plots the left and right polynomials on the lane lines
    if debug:
        plt.plot(left_fitx, ploty, color='red')
        plt.plot(right_fitx, ploty, color='blue')
        plt.imshow(out_img, cmap='gray')
        plt.show()

    return left_fit, right_fit, left_fitx, right_fitx, ploty, out_img


# def put_points(image, pts, color=[0, 255, 0]):
#     # pdb.set_trace()
#     for (x, y) in pts:
#         x = int(x)
#         y = int(y)
#         print(x, y, color)
#         image[y, x, :] = color
#
#     pdb.set_trace()
#     return image


def draw_lane(image, coeffs, color=(255, 0, 0), line_width=50):
    """
    Draw a second degree polynomial
    """
    h, w, c = image.shape
    ploty = np.linspace(0, h - 1, h)

    line_center = coeffs[0] * ploty ** 2 + coeffs[1] * ploty + coeffs[2]
    line_left_side = line_center - line_width // 2
    line_right_side = line_center + line_width // 2

    pts_left = np.array(list(zip(line_left_side, ploty)))
    pts_right = np.array(np.flipud(list(zip(line_right_side, ploty))))
    pts = np.vstack([pts_left, pts_right])

    # Draw the lane onto the warped blank image
    return cv2.fillPoly(image, [np.int32(pts)], color)


def add_lane(image, warped, Minv, left_fit, right_fit, left_fitx, right_fitx, ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Draw green region
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Draw lane lines
    draw_lane(color_warp, left_fit, color=(255, 0, 0))
    draw_lane(color_warp, right_fit, color=(0, 0, 255))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    top_layer = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    result = cv2.addWeighted(image, 1, top_layer, 0.3, 0)

    return result



def generate_data(leftx, rightx, ploty):
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    return ploty, left_fit_cr, right_fit_cr


def measure_curvature_real(leftx, rightx, ploty):
    # Define conversions in x and y from pixels space to meters

    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    ploty, left_fit_cr, right_fit_cr = generate_data(leftx, rightx, ploty)

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return np.average([left_curverad, right_curverad])



def compute_offset(birdeye, left_fitx, right_fitx, ploty):
    image_center = birdeye.shape[1] / 2

    left_bottom = np.mean(left_fitx[ploty > 0.95 * max(ploty)])
    right_bottom = np.mean(right_fitx[ploty > 0.95 * max(ploty)])
    lane_width = right_bottom - left_bottom
    lane_center = (left_bottom + lane_width / 2)

    offset_pix = lane_center - image_center
    offset_meter = xm_per_pix * offset_pix

    return offset_meter


if __name__ == '__main__':
    # image = mpimg.imread("test_images/test2.jpg")
    image = mpimg.imread("test_images/test1.jpg")
    # image = mpimg.imread("test_images/straight_lines2.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # calibrate camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera()
    calibrated_image = undistort(image, mtx, dist, debug=False)

    # color binary
    color_binary = compute_color_binary(calibrated_image, debug=False)

    # warp
    warp_binary, M, Minv = warp_to_birdeye(color_binary, debug=False)


    left_fit, right_fit, left_fitx, right_fitx, ploty, birdeye_lane = fit_polynomial(warp_binary)


    result = add_lane(calibrated_image, warp_binary, left_fit, right_fit, left_fitx, right_fitx, ploty)


    # Calculate the radius of curvature in pixels for both lane lines
    # left_curverad, right_curverad = measure_curvature_pixels(left_fit, right_fit, ploty)
    # left_curverad, right_curverad = measure_curvature_real(left_fit, right_fit, ploty)
    curvature = measure_curvature_real(left_fitx, right_fitx, ploty)

    print(curvature)
    # Should see values of 1625.06 and 1976.30 here, if using
    # the default `generate_data` function with given seed number

    compute_offset(warp_binary, left_fitx, right_fitx, ploty)


    # show images
    color_binary_image = binary_to_image(color_binary)
    warp_image = binary_to_image(warp_binary)
    frame = add_info(result, color_binary_image, warp_image, birdeye_lane)
    show_image(frame)


