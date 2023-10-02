import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy import ndimage
from scipy.signal import find_peaks

DEBUG = True
DEBUG_OUT_PATH = "./debug/"

if not os.path.exists(DEBUG_OUT_PATH):
    os.makedirs(DEBUG_OUT_PATH)
elif os.path.exists(DEBUG_OUT_PATH):
    for file in os.listdir(DEBUG_OUT_PATH):
        os.remove(os.path.join(DEBUG_OUT_PATH, file))


def save_img(img: np.ndarray | list, filename: str):
    """Save the image to the specified path."""

    if isinstance(img, list):
        img = np.array(img)
    cv2.imwrite(os.path.join(DEBUG_OUT_PATH, filename), img)


def preprocess(img: np.ndarray, height, width):
    """Run the preprocessing steps on the image.

    Convert to grayscale, threshold, blur, and find circles."""

    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if DEBUG:
        save_img(bw_img, "bw.bmp")

    _, thresh_img = cv2.threshold(bw_img, 20, 255, cv2.THRESH_BINARY)

    if DEBUG:
        save_img(thresh_img, "thresh.bmp")

    blur_img = cv2.blur(thresh_img, (5, 5))

    if DEBUG:
        save_img(blur_img, "blur.bmp")

    # Attempt to find the main lock circle
    main_lock_circle: list = cv2.HoughCircles(
        blur_img, cv2.HOUGH_GRADIENT, 1, height * 2, param1=100, param2=30, minRadius=100, maxRadius=width
    )

    if DEBUG:
        # Convert the circle parameters a, b and r to integers.
        main_lock_circle = np.uint16(np.around(main_lock_circle))
        lock_img = img.copy()

        for pt in main_lock_circle[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(lock_img, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(lock_img, (a, b), 1, (0, 0, 255), 3)

        save_img(lock_img, "lock_circles.bmp")

    return bw_img, thresh_img, blur_img, main_lock_circle


def reproject(img, center_x, center_y, radius, num_rays, base_img):
    """Reproject the image to a flat plane.

    img: the sampled image
    center_x: the x coordinate of the center of the lock
    center_y: the y coordinate of the center of the lock
    radius: the radius of the lock
    num_rays: the number of rays to sample

    """
    new_img = []
    sample_copy = img.copy()
    sample_copy = cv2.cvtColor(sample_copy, cv2.COLOR_GRAY2RGB)

    # retreat the radius inward until we just have black pixels
    for r in range(radius, 0, -2):
        pixel_vals = []

        for i in range(num_rays):
            theta = i * (360 / num_rays)
            rad = np.deg2rad(theta)

            x = int(center_x + r * np.cos(rad))
            y = int(center_y + r * np.sin(rad))

            pixel_vals.append(img[y, x])
            # if DEBUG:
            # dot_update = cv2.circle(sample_copy, (x, y), 1, (0, 0, 255), 2)

        pix_count = np.count_nonzero(pixel_vals)
        # cv2.circle(sample_copy, (center_x, center_y), r, (0, 255, 0), 2)
        # save_img(sample_copy, "radius_retreat.bmp")
        if pix_count == 0:
            radius = r
            break

    for i in range(num_rays):
        theta = i * (360 / num_rays)
        rad = np.deg2rad(theta)

        r_vals = []

        for r in range(0, radius, 1):
            x = int(center_x + r * np.cos(rad))
            y = int(center_y + r * np.sin(rad))

            # Draw the sample dots for debugging.
            # if DEBUG:
            # dot_update = cv2.circle(sample_copy, (x, y), 1, (0, 255, 0), 2)

            r_vals.append(img[y, x])

        new_img.append(r_vals)

    new_img = np.array(new_img)

    # if DEBUG:
    #     save_img(dot_update, "dot_update.bmp")
    save_img(new_img, "reprojected_img.bmp")

    return new_img


def truncate_edge(img_array: np.ndarray):
    """Iterate over the rightmost pixels of the image. If any of the pixels are white
    remove the entire column from the array.
    """

    for i in range(img_array.shape[1] - 1, 0, -1):
        if np.any(img_array[:, i] == 255):
            img_array = np.delete(img_array, i, 1)
        else:
            break

    save_img(img_array, "truncated.bmp")
    return img_array


def align_outputs(img_array: np.ndarray):
    """Cross-correlate each line of the image array and output a new image
    with the lines aligned.
    """

    aligned_img = np.zeros(img_array.shape)
    line_avgs = np.zeros(img_array.shape)

    for i in range(img_array.shape[0]):
        if i == 0:
            line_avgs[i, :] = (img_array[-1, :] + img_array[i, :] + img_array[i + 1, :]) / 3
        elif i == img_array.shape[0] - 1:
            line_avgs[i, :] = (img_array[i - 1, :] + img_array[i, :] + img_array[0, :]) / 3
        else:
            line_avgs[i, :] = (img_array[i - 1, :] + img_array[i, :] + img_array[i + 1, :]) / 3

    if DEBUG:
        save_img(line_avgs, "line_avgs.bmp")

    # save max lag list to align averages later
    # use this for centering the final gap sampling
    lag_list = []

    for i in range(line_avgs.shape[0]):
        for j in range(line_avgs.shape[1]):
            if line_avgs[i, j] != 0:
                break

        # roll the main image array by the offset
        aligned = np.roll(img_array[i, :], -j)
        aligned_img[i, :] = aligned
        lag_list.append(j)
    if DEBUG:
        save_img(aligned_img, "aligned.bmp")

    # do the same roll to every line of the averages
    for i in range(line_avgs.shape[0]):
        line_avgs[i, :] = np.roll(line_avgs[i, :], -lag_list[i])

    if DEBUG:
        save_img(line_avgs, "aligned_avgs.bmp")

    return aligned_img, line_avgs


def get_peaks(avg_image: np.ndarray):
    """Find the peaks of the averaged image. Take the average of the peaks
    row-wise to find the center of each lock ring."""

    peaks = []

    for i in range(avg_image.shape[0]):
        img_row = avg_image[i, :]
        img_row = ndimage.uniform_filter1d(img_row, 5)

        local_peaks = []
        state = False

        # move along the row, if the value is greater than ~20, enter a high state
        # this is the start of the peak
        # when the value drops below 20, enter a low state
        # record the start and end of the peaks and take the average

        for j in range(img_row.shape[0]):
            if img_row[j] > 20 and not state:
                state = True
                start = j
            elif img_row[j] < 20 and state:
                state = False
                end = j
                local_peaks.append(int((start + end) / 2))

        peaks.append(local_peaks)

    # move down the peaks list and take the average of every peak
    peaks = np.mean(peaks, axis=0).astype(int)

    return peaks


def sample_ring_gaps(aligned_img: np.ndarray, peaks):
    """Use the peaks to sample the gaps between the rings."""

    peak_samples = []
    for row in range(aligned_img.shape[0]):
        row_samples = []
        for peak in peaks:
            row_samples.append(aligned_img[row, peak])
        peak_samples.append(row_samples)

    # flip the image left to right
    peak_samples = np.flip(peak_samples, axis=1)

    if DEBUG:
        save_img(peak_samples, "peak_samples.bmp")

    return peak_samples