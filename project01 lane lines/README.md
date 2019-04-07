
## Code

- [code](main.py)

- [image outputs](test_images_output)

- [video outputs](test_videos_output)


## Image PipeLine
The code is in `main#process_image`

- Grayscale
- gaussian blue
- Canny edge detection
- Mask polygon
- Hough line detection
- Split Hough lines into left and right into two group of points. Splitting
  decision is based on each line's slope.
- For each group of points, fit a line by least square.
- Draw the line with boundary conditions


## Video PipeLine
The code is in `main#get_marked_frame`

- Keep a buffer up to 7 frames
- For these frames, repeat the image pipeline up to Hough line detection.
- Aggregate all the Hough lines from each frame
- Split Hough lines into left and right into two group of points. Splitting
  decision is based on each line's slope.
- For each group of points, fit a line by least square.
- Draw the line with boundary conditions


## Shortcoming and Improvements
- The code was not optimized for the challenge.mp4
- Edge detection could be thrown off if the masked region has fast gradient changes
- The lane lines could be extended further into the horizon
- The lane lines could be fitted into a curve instead of straight lines
