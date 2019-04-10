import logging
import os

import cv2
import matplotlib.image as mpimg
import numpy as np

from camera import calibrate_camera, undistort, warp_to_birdeye
from color import compute_color_binary
from lane import add_lane, fit_polynomial, measure_curvature_real2, compute_offset2
from sequence import initialize_state, process_frame
from utils import show_image, add_info, binary_to_image


# Logging
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
    # calibration
    ret, mtx, dist, rvecs, tvecs = calibrate_camera()
    calibrated_image = undistort(image, mtx, dist, debug=False)

    # color binary
    color_binary = compute_color_binary(calibrated_image, debug=False)

    # warp
    birdeye_binary, M, Minv = warp_to_birdeye(color_binary, debug=False)

    # lane
    left_fit, right_fit, left_fitx, right_fitx, ploty, birdeye_lane = fit_polynomial(birdeye_binary, debug=False)

    # curvature
    curvature = measure_curvature_real2(calibrated_image, birdeye_binary, Minv,
                                        np.int_(left_fitx),
                                        np.int_(right_fitx),
                                        np.int_(ploty))

    # offset
    offset_meter = compute_offset2(calibrated_image, birdeye_binary, Minv,
                                   np.int_(left_fitx),
                                   np.int_(right_fitx),
                                   np.int_(ploty))

    # Draw the line and green region
    result = add_lane(calibrated_image, birdeye_binary, Minv, left_fit, right_fit, left_fitx, right_fitx, ploty)

    # Information overlay
    text_data = {
        "Curvature": f"{curvature:.0f} meter",
        "Offset": f"{offset_meter:.2f} meter from center",
    }
    color_binary_image = binary_to_image(color_binary)
    warp_image = binary_to_image(birdeye_binary)
    frame = add_info(result, color_binary_image, warp_image, birdeye_lane, text_data)

    # Debug: show image
    if debug:
        show_image(frame)

    return frame


def process_video_file(filename):
    input_filename = f'test_videos/{filename}'
    output_filename = f'output_videos/{filename}'
    logger.info(f'Processing video: {input_filename}')
    logger.info(f'Output videos: {output_filename}')

    state = initialize_state()

    resize_h, resize_w = 720, 1280
    cap = cv2.VideoCapture(input_filename)
    out = cv2.VideoWriter(output_filename,
                          fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                          fps=25.0,
                          frameSize=(resize_w, resize_h))

    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (resize_w, resize_h))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Debug message
            counter += 1
            logger.info(f"{counter}: {frame.shape}")

            # Get the frame with overlay information
            marked_frame = process_frame(state, frame, debug=False)

            marked_frame = cv2.cvtColor(marked_frame, cv2.COLOR_RGB2BGR)
            out.write(marked_frame)
        else:
            break
    cap.release()
    out.release()


def main():
    # Processing images
    for filename in os.listdir("test_images/"):
        process_image_file(filename)

    process_video_file("project_video.mp4")
    # process_video_file("challenge_video.mp4")
    # process_video_file("harder_challenge_video.mp4")


if __name__ == '__main__':
    main()
