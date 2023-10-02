from itertools import product

import cv2
import numpy as np

from utils.img_utils import save_img


def refine_center(center_x, center_y, radius, img):
    """Refine the center of the lock.

    Optimally, the center of the lock will be where the majority of pixels at
    the radius of the lock will be white.

    center_x: the x coordinate of the center of the lock
    center_y: the y coordinate of the center of the lock
    radius: the radius of the lock
    img: the image to refine the center of

    """

    # All of this best center thing doesn't really work that well

    best_center = (center_x, center_y)
    best_radius = radius

    PADDING = 20
    num_rays = 32

    x_range = range(center_x - PADDING, center_x + PADDING)
    y_range = range(center_y - PADDING, center_y + PADDING)
    radius_range = range(radius - PADDING, min(radius + PADDING, img.shape[0], img.shape[1]))

    opt_combos = list(product(x_range, y_range))

    opt_combos.insert(0, (center_x, center_y))

    for x, y in opt_combos:
        found = False
        for r in radius_range:
            pixel_count, pixels = sample_ring_pixels(x, y, r, num_rays, img)

            if pixel_count == num_rays:
                best_center = (y, x)
                best_radius = r
                found = True
                break
        if found:
            break
    better_fit_img = img.copy()

    # Draw the circumference of the circle.
    better_fit_img = cv2.cvtColor(better_fit_img, cv2.COLOR_GRAY2RGB)
    cv2.circle(better_fit_img, best_center, best_radius, (0, 255, 0), 2)

    for i in range(num_rays):
        theta = i * (360 / num_rays)
        rad = np.deg2rad(theta)

        edge_x = int(best_center[0] + best_radius * np.cos(rad))
        edge_y = int(best_center[1] + best_radius * np.sin(rad))
        cv2.circle(better_fit_img, (edge_x, edge_y), 1, (0, 0, 255), 3)

        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(better_fit_img, best_center, 1, (0, 0, 255), 3)

        save_img(better_fit_img, "better_fit.bmp")

    return best_center, best_radius


def sample_ring_pixels(center_x, center_y, radius, num_rays, img):
    """Sample a ring of pixels from the image."""

    pixel_vals = []

    for i in range(num_rays):
        theta = i * (360 / num_rays)
        rad = np.deg2rad(theta)

        x = int(center_x + radius * np.cos(rad))
        y = int(center_y + radius * np.sin(rad))

        try:
            pixel_vals.append(img[x, y])
        except IndexError:
            pixel_vals.append(0)

    pix_count = np.count_nonzero(pixel_vals)

    return pix_count, pixel_vals
