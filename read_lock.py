import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """
    # define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32) / dY.astype(np.float32)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(int) + P1X
        else:
            slope = dY.astype(np.float32) / dX.astype(np.float32)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(int) + P1Y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

    return itbuffer


base_img = cv2.imread("lock2.png", cv2.IMREAD_COLOR)
img = base_img.copy()

# get the dimensions of the image
height, width, colors = img.shape

cv2.imwrite("lock_cropped.jpg", img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
cv2.imwrite("lock_threshold.jpg", img)

img = cv2.blur(img, (4, 4))
cv2.imwrite("lock_blur.jpg", img)

detected_circles: list = cv2.HoughCircles(
    img, cv2.HOUGH_GRADIENT, 1, height, param1=100, param2=30, minRadius=100, maxRadius=width
)


if detected_circles is not None:
    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))

    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]

        # Draw the circumference of the circle.
        cv2.circle(base_img, (a, b), r, (0, 255, 0), 2)

        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(base_img, (a, b), 1, (0, 0, 255), 3)
        cv2.imwrite("lock_circles.jpg", base_img)

outer_lock_circle = detected_circles[0]
center_x, center_y, lock_radius = outer_lock_circle[0][0], outer_lock_circle[0][1], outer_lock_circle[0][2]

ray_vals = []

for i in range(32):
    # draw 32 lines around the center
    angle = i * 11.25
    rad = np.deg2rad(angle)
    end_x = int(center_x + lock_radius * np.cos(rad))
    end_y = int(center_y + lock_radius * np.sin(rad))

    # extract the values of the pixels under the line
    line_iter = createLineIterator((center_x, center_y), (end_x, end_y), img)
    ray_vals.append(line_iter[:, 2])

    # draw the line
    cv2.line(base_img, (center_x, center_y), (end_x, end_y), (255, 255, 255, 255), 2)
    # label the line
    cv2.putText(
        base_img,
        str(i),
        (int(end_x * 1.01), int(end_y * 1.01)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite("lock_lines.jpg", base_img)

# for each ray, find the peaks
peaks = []
for ray in ray_vals:
    ray_peaks, _ = find_peaks(ray, height=250)
    # limit to the first 4 peaks
    peaks.append(ray_peaks[:4])


def draw_peak_locations(img, peaks):
    for i, ray in enumerate(peaks):
        angle = i * 11.25
        rad = np.deg2rad(angle)
        for peak in ray:
            end_x = int(center_x + peak * np.cos(rad))
            end_y = int(center_y + peak * np.sin(rad))

            cv2.circle(img, (end_x, end_y), 3, (0, 0, 255), 2)

            cv2.putText(
                img,
                str(peak),
                (int(end_x * 1.01), int(end_y * 1.01)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite("lock_peaks.jpg", img)


draw_peak_locations(base_img, peaks)

print(peaks)