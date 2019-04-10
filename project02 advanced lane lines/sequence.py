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

from moviepy.editor import VideoFileClip



# # Imports for Jupyter
# %matplotlib inline

import logging

from camera import calibrate_camera, undistort, warp_to_birdeye
from color import compute_color_binary
from lane import add_lane, measure_curvature_real2, \
    compute_offset2, find_lane_pixels, fit_polynomial, draw_lane
from utils import show_image, add_info, binary_to_image

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_state():
    ret, mtx, dist, rvecs, tvecs = calibrate_camera()
    state = {
        'ret': ret,
        'mtx': mtx,
        'dist': dist,
        'rvecs': rvecs,
        'tvecs': tvecs,
    }

    return state


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fitx, right_fitx, ploty


def search_around_poly(state, birdeye_binary, debug=False):
    left_fit = state['left_fit']
    right_fit = state['right_fit']

    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = birdeye_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                                                                         left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                                                                           right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    out_img = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary))*255

    if debug:
        # Fit new polynomials
        left_fitx, right_fitx, ploty = fit_poly(birdeye_binary.shape, leftx, lefty, rightx, righty)

        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Plot the polynomial lines onto the image
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        ## End visualization steps ##

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial_with(state, birdeye_binary, debug=False):
    # First fit
    if 'left_fit' not in state:
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(birdeye_binary)
        left_fit, right_fit, left_fitx, right_fitx, ploty, out_img = fit_polynomial(birdeye_binary, debug=False)

        import collections
        que_size = 5
        state['leftx_deque'] = collections.deque(maxlen=que_size)
        state['lefty_deque'] = collections.deque(maxlen=que_size)
        state['rightx_deque'] = collections.deque(maxlen=que_size)
        state['righty_deque'] = collections.deque(maxlen=que_size)

        state['leftx_deque'].append(leftx)
        state['lefty_deque'].append(lefty)
        state['rightx_deque'].append(rightx)
        state['righty_deque'].append(righty)
        state['left_fit'] = left_fit
        state['right_fit'] = right_fit

        return left_fit, right_fit, left_fitx, right_fitx, ploty, out_img


    # Subsequent fit
    _leftx, _lefty, _rightx, _righty, out_img = search_around_poly(state, birdeye_binary, debug=False)

    leftx = np.copy(_leftx)
    lefty = np.copy(_lefty)
    rightx = np.copy(_rightx)
    righty = np.copy(_righty)

    for more in state['leftx_deque']:
        leftx = np.concatenate((leftx, more))
    for more in state['lefty_deque']:
        lefty = np.concatenate((lefty, more))
    for more in state['rightx_deque']:
        rightx = np.concatenate((rightx, more))
    for more in state['righty_deque']:
        righty = np.concatenate((righty, more))

    # leftx, lefty, rightx, righty
    # pdb.set_trace()

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, birdeye_binary.shape[0]-1, birdeye_binary.shape[0] )

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
        # plt.plot(left_fitx, ploty, color='red')
        # plt.plot(right_fitx, ploty, color='blue')
        plt.imshow(out_img, cmap='gray')
        plt.show()

    # Fix state
    state['leftx_deque'].append(_leftx)
    state['lefty_deque'].append(_lefty)
    state['rightx_deque'].append(_rightx)
    state['righty_deque'].append(_righty)
    state['left_fit'] = left_fit
    state['right_fit'] = right_fit

    return left_fit, right_fit, left_fitx, right_fitx, ploty, out_img


def process_frame(state, frame, debug=False):
    ret = state['ret']
    mtx = state['mtx']
    dist = state['dist']
    rvecs = state['rvecs']
    tvecs = state['tvecs']

    # camera calibration
    calibrated_image = undistort(frame, mtx, dist, debug=False)

    # color binary
    color_binary = compute_color_binary(calibrated_image, debug=False)

    # birdeye
    birdeye_binary, M, Minv = warp_to_birdeye(color_binary, debug=False)

    # lane
    left_fit, right_fit, left_fitx, right_fitx, ploty, birdeye_lane = fit_polynomial_with(state, birdeye_binary, debug=debug)

    curvature = measure_curvature_real2(calibrated_image, birdeye_binary, Minv,
                                        np.int_(left_fitx),
                                        np.int_(right_fitx),
                                        np.int_(ploty))
    offset_meter = compute_offset2(calibrated_image, birdeye_binary, Minv,
                                   np.int_(left_fitx),
                                   np.int_(right_fitx),
                                   np.int_(ploty))
    text_data = {
        "Curvature": f"{curvature:.0f} meter",
        "Offset": f"{offset_meter:.2f} meter from center",
    }

    result = add_lane(calibrated_image, birdeye_binary, Minv, left_fit, right_fit, left_fitx, right_fitx, ploty)

    color_binary_image = binary_to_image(color_binary)
    warp_image = binary_to_image(birdeye_binary)
    frame = add_info(result, color_binary_image, warp_image, birdeye_lane, text_data)

    # show images
    if debug:
        show_image(frame)

    return frame


def make_subclip():
    clip = VideoFileClip("test_videos/project_video.mp4")
    subclip = clip.subclip(23, 26)
    subclip.write_videofile("test_videos/small23-26.mp4")


def grab_frame(num):
    filename = 'project_video.mp4'
    input_filename = f'test_videos/{filename}'
    logger.info(f'Reading video: {input_filename}')

    resize_h, resize_w = 720, 1280
    cap = cv2.VideoCapture(input_filename)
    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (resize_w, resize_h))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            counter += 1
            print(counter, frame.shape)

            if counter == num:
                mpimg.imsave(f"test_images/{num}.jpg", frame)
                break
        else:
            break
    cap.release()


def main():
    # # Processing images
    # for filename in os.listdir("test_images/"):
    #     process_image_file(filename)

    # process_image_file("613.jpg", debug=True)
    # process_image_file("test1.jpg")

    # process_video_file("project_video.mp4")
    # process_video_file("small.mp4")
    # process_video_file("small24-25.mp4")
    # process_video_file("small23-26.mp4")
    pass


if __name__ == '__main__':
    main()

    # make_subclip()

    # import timeit
    # print(timeit.timeit("main()", setup="from __main__ import main", number=1))



# # import cv2
# if __name__ == '__main__' :
#
#     video = cv2.VideoCapture("test_videos/small.mp4");
#
#     # Find OpenCV version
#     (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
#
#     if int(major_ver)  < 3 :
#         fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
#         print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
#     else :
#         fps = video.get(cv2.CAP_PROP_FPS)
#         print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
#
#     video.release();
