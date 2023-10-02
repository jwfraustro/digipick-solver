import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.img_utils import align_outputs, get_peaks, preprocess, reproject, sample_ring_gaps, save_img, truncate_edge
from utils.optimizer import refine_center


def solve_lock_dfs(rings, picks, lock_index, used_picks=[]):
    """Perform a depth-first search to solve the lock."""

    solve_str = ""

    if lock_index == len(rings):
        return " - SOLVED"

    if np.all(rings[lock_index]):
        # we've solved this one, move on
        solve_return = solve_lock_dfs(rings, picks, lock_index + 1, used_picks)
        if solve_return:
            return solve_str + solve_return
    elif len(used_picks) == len(picks):
        return "No picks left."

    # check to see if any picks left have the correct number of points remaining
    valid = False
    for i, pick in enumerate(picks):
        if len(np.argwhere(pick == True)) <= len(np.argwhere(rings[lock_index] == False)):
            valid = True
    if not valid:
        return "No picks left."

    for pick_num, pick in enumerate(picks):
        if pick_num in used_picks:
            continue
        for angle in range(0, 32):
            # roll the pick and insert it into the ring
            rolled_pick = np.roll(pick, angle)
            insert = rings[lock_index][np.argwhere(rolled_pick == True)]

            # no match
            if np.any(insert):
                continue

            # make a copy of the remaining rings
            remaining_rings = rings.copy()
            # fill the ring with the pick
            remaining_rings[lock_index] = remaining_rings[lock_index] + rolled_pick
            used_picks_copy = [x for x in used_picks]
            used_picks_copy.append(pick_num)

            solve_ret = solve_lock_dfs(remaining_rings, picks, lock_index, used_picks_copy)
            if solve_ret:
                if angle > 16:
                    angle = angle - 32
                return solve_str + f"{pick_num}[{angle}]" + solve_ret

    return solve_str + f"{pick_num}[{angle}]"


def main():
    IMG_PATH = "./test_imgs/rl.png"


    base_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    img = base_img.copy()

    # get the dimensions of the image
    height, width, colors = img.shape

    bw_img, thresh_img, blur_img, detected_circles = preprocess(img, height, width)

    outer_lock_circle = detected_circles[0]
    center_x, center_y, lock_radius = outer_lock_circle[0][0], outer_lock_circle[0][1], outer_lock_circle[0][2]

    # (center_x, center_y), lock_radius = refine_center(center_x, center_y, lock_radius, thresh_img)

    reprojected_img = reproject(
        thresh_img,
        center_x,
        center_y,
        lock_radius,
        32,
        base_img,
    )

    # trunc_img = truncate_edge(reprojected_img)
    aligned_img, line_averages = align_outputs(reprojected_img)
    ring_peaks = get_peaks(line_averages)
    peak_samples = sample_ring_gaps(aligned_img, ring_peaks)

    # convert peak samples to a boolean array
    lock_bools = np.array(peak_samples) > 0

    # look for the picks in the image
    pick_max_radius = int(0.15 * lock_radius)
    pick_min_radius = int(0.09 * lock_radius)

    bw_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(bw_img, 20, 255, cv2.THRESH_BINARY)
    blur_img = cv2.blur(thresh_img, (2, 2))
    cv2.imwrite("blur.bmp", blur_img)
    pick_circles: list = cv2.HoughCircles(
        blur_img,
        cv2.HOUGH_GRADIENT,
        1,
        pick_max_radius,
        param1=100,
        param2=30,
        minRadius=pick_min_radius,
        maxRadius=pick_max_radius,
    )

    pick_circles = np.uint16(np.around(pick_circles))

    pick_vals = []

    for pick_num, pt in enumerate(pick_circles[0, :]):
        a, b, r = pt[0], pt[1], pt[2]

        cv2.circle(base_img, (a, b), 1, (0, 0, 255), 3)
        cv2.circle(base_img, (a, b), r, (0, 255, 0), 2)
        # label the pick
        cv2.putText(base_img, f"{pick_num}", (a + 5, b + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imwrite("pick_circles.bmp", base_img)

        num_rays = 32

        new_img = []
        r_vals = np.full(num_rays, 255)

        while np.count_nonzero(r_vals) > 4:
            for i in range(num_rays):
                theta = i * (360 / num_rays)
                rad = np.deg2rad(theta)

                x = int(a + r * np.cos(rad))
                y = int(b + r * np.sin(rad))

                # Draw the sample dots for debugging.
                # if DEBUG:
                # dot_update = cv2.circle(sample_copy, (x, y), 1, (0, 255, 0), 2)

                r_vals[i] = thresh_img[y, x]
            r -= 1

        if np.count_nonzero(r_vals) > 0:
            new_img = np.array(r_vals)
            pick_vals.append(new_img)
            # cv2.circle(base_img, (a, b), r, (0, 255, 0), 2)
            # save_img(new_img, f"reprojected_{pick_num}_img.bmp")
        else:
            warnings.warn(f"Pick {pick_num} doesn't have any points.")

    pick_bools = np.array(pick_vals) > 0

    lock_bools = lock_bools.transpose()

    all_rings = lock_bools
    all_picks = pick_bools

    for pick in all_picks:
        print(np.count_nonzero(pick))

    print(solve_lock_dfs(all_rings, all_picks, 0))


if __name__ == "__main__":
    main()