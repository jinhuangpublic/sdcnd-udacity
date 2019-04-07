import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
from collections import deque
import numpy as np
from moviepy.editor import VideoFileClip
from os.path import join, basename
# from moviepy.editor import VideoFileClip


# # Imports for Jupyter
# %matplotlib inline

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def jupyter():
    try:
        # noinspection PyUnresolvedReferences
        get_ipython()
        return True
    except NameError:
        return False


def display_gray(image):
    logger.info(image.shape)
    plt.figure()
    plt.imshow(image, cmap='gray')
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()


def display_color(image):
    logger.info(image.shape)
    plt.figure()
    plt.imshow(image)


def get_image(filename):
    return mpimg.imread(filename)


def crop(image):
    y = image.shape[0]
    x = image.shape[1]
    vertices = np.array([[
        (0.1*x, y),
        (0.45*x, 0.6*y),
        (0.55*x, 0.6*y),
        (0.95*x, y)
    ]], dtype=np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(image, mask)

    return masked_edges


class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __repr__(self):
        return str([self.x1, self.y1, self.x2, self.y2])

    def get_slope(self):
        return (self.y2 - self.y1) / (self.x2 - self.x1)


def get_lines(image):
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi/180  # angular resolution in radians of the Hough grid
    threshold = 1   # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 60  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(image,
                            rho,
                            theta,
                            threshold,
                            None,
                            min_line_length,
                            max_line_gap)

    out = []
    if lines is not None:
        for l in lines:
            out.append(Line(l[0][0], l[0][1], l[0][2], l[0][3]))
    return out


def fit_lane_lines(lines, image_shape):
    _lines = []

    left_line = compute_line(lines, image_shape, -5, -0.2)
    if left_line is not None:
        _lines.append([left_line])

    right_line = compute_line(lines, image_shape, 0.2, 5)
    if right_line is not None:
        _lines.append([right_line])

    return _lines


def compute_line(lines, image_shape, slope_lower, slope_up):
    x = []
    y = []
    min_x = image_shape[0]
    for line in lines:
        if slope_lower < line.get_slope() < slope_up:
            x.append(line.y1)
            x.append(line.y2)
            y.append(line.x1)
            y.append(line.x2)
            min_x = min([min_x, line.y1, line.y2])
    if x:
        p = np.polyfit(x, y, 1)
        x1, y1 = min_x, np.polyval(p, min_x)
        x2, y2 = image_shape[0], np.polyval(p, image_shape[0])
        return np.array([y1, x1, y2, x2], dtype=np.int32)
    else:
        return None


def draw_lane_lines(image, lines, color=[255, 51, 51], thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)


def extract_lines(image):
    # Grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    image = cv2.GaussianBlur(image, (3, 3), 0)  # 17

    # Detect edges
    image = cv2.Canny(image, threshold1=50, threshold2=150)

    # Crop
    image = crop(image)

    # Detect lines
    lines = get_lines(image)
    return lines


def get_marked_frame(buffer):
    """
    Draw a line on the last frame based on a short history of frames.
    """
    frame = buffer[-1]
    lines = []
    for image in buffer:
        lines = lines + extract_lines(image)

    lines = fit_lane_lines(lines, frame.shape)
    draw_lane_lines(frame, lines, color=[73, 73, 255], thickness=7)

    return frame


def grab_frame_from_video(videoname, t=1):
    input_filename = f'test_videos/{videoname}'
    clip = VideoFileClip(input_filename)
    frame = clip.get_frame(t)
    return frame


def process_image(image):
    lines = extract_lines(image)
    lane_lines = fit_lane_lines(lines, image.shape)
    draw_lane_lines(image, lane_lines)
    return image


def process_image_file(filename):
    input_filename = f'test_images/{filename}'
    output_filename = f'test_images_output/{filename}'

    logger.info(f'Processing image: {input_filename}')

    image = get_image(f'test_images/{filename}')
    image = process_image(image)
    mpimg.imsave(output_filename, image)

    if jupyter():
        display_color(image)


def process_video_file(filename):
    input_filename = f'test_videos/{filename}'
    output_filename = f'test_videos_output/{filename}'

    logger.info(f'Processing video: {input_filename}')

    resize_h, resize_w = 540, 960
    cap = cv2.VideoCapture(input_filename)
    out = cv2.VideoWriter(output_filename,
                          fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                          fps=20.0,
                          frameSize=(resize_w, resize_h))

    buffer = deque(maxlen=7)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (resize_w, resize_h))
            buffer.append(frame)
            marked_frame = get_marked_frame(buffer)
            out.write(marked_frame)
        else:
            break
    cap.release()
    out.release()


def main():
    # Processing images
    for filename in os.listdir("test_images/"):
        process_image_file(filename)

    # Processing videos
    for videoname in os.listdir("test_videos/"):
        process_video_file(videoname)

        # Testing only: grab one frame
        # frame = grab_frame_from_video(videoname, 10)
        # process_image(frame)


if __name__ == '__main__':
    main()
