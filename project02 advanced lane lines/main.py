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
from lane import measure_curvature_real, add_lane, fit_polynomial, compute_offset
from utils import show_image, add_info, binary_to_image

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def process_image_file(filename, debug=False):
    input_filename = f'test_images/{filename}'
    output_filename = f'output_images/{filename}'
    logger.info(f'Processing image: {input_filename}')

    image = mpimg.imread(input_filename)
    image = process_image(image, debug)
    mpimg.imsave(output_filename, image)


def process_image(image, debug=False):

    ret, mtx, dist, rvecs, tvecs = calibrate_camera()
    calibrated_image = undistort(image, mtx, dist, debug=False)

    # color binary
    color_binary = compute_color_binary(calibrated_image, debug=False)

    # warp
    birdeye_binary, M, Minv = warp_to_birdeye(color_binary, debug=False)

    # lane
    left_fit, right_fit, left_fitx, right_fitx, ploty, birdeye_lane = fit_polynomial(birdeye_binary, debug=False)

    curvature = measure_curvature_real(left_fitx, right_fitx, ploty)
    offset_meter = compute_offset(birdeye_binary, left_fitx, right_fitx, ploty)
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


def process_video_file_simple(filename):
    input_filename = f'test_videos/{filename}'
    output_filename = f'output_videos/{filename}'

    logger.info(f'Processing video: {input_filename}')
    logger.info(f'Output videos: {output_filename}')

    resize_h, resize_w = 720, 1280
    cap = cv2.VideoCapture(input_filename)
    out = cv2.VideoWriter(output_filename,
                          fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                          fps=25.0,
                          frameSize=(resize_w, resize_h))

    # out = cv2.VideoWriter(output_filename,
    #                       fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
    #                       fps=20.0,
    #                       frameSize=(resize_w, resize_h))

    # buffer = deque(maxlen=7)
    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # frame = cv2.resize(frame, (resize_w, resize_h))
            # print(counter, frame.shape)
            # out.write(frame)

            frame = cv2.resize(frame, (resize_w, resize_h))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            marked_frame = process_image(frame)
            # marked_frame = frame

            counter += 1
            # if marked_frame is None:
            print(counter, marked_frame.shape)

            marked_frame = cv2.cvtColor(marked_frame, cv2.COLOR_RGB2BGR)


            out.write(marked_frame)
        else:
            break
    cap.release()
    out.release()



def make_subclip():
    clip = VideoFileClip("test_videos/project_video.mp4")
    subclip = clip.subclip(0, 3)
    subclip.write_videofile("test_videos/small.mp4")


def main():
    # Processing images
    # for filename in os.listdir("test_images/"):
    #     process_image_file(filename)

    # process_image_file("straight_lines1.jpg", debug=True)
    # process_image_file("test1.jpg")

    process_video_file_simple("small.mp4")



if __name__ == '__main__':
    main()

    # make_subclip()



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
