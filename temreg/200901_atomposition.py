import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils

checkBlur = False


def blur_subtract(array, kernel_size=10):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    dst = cv2.filter2D(array, -1, kernel)

    diff = cv2.subtract(array, dst)
    return diff


imagePath = "IFFT of FFT of PtCo_IL_b-0032_NS.tif"
image = cv2.imread(imagePath, 0)
image_rgb = cv2.imread(imagePath, 1)

if checkBlur:
    # setup the figure
    fig = plt.figure("Original vs. blurred")

    # show first image
    fig.add_subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.axis("off")

    # show the second image
    fig.add_subplot(1, 4, 2)
    plt.imshow(blur_subtract(image, 10), cmap='gray')
    plt.axis("off")

    fig.add_subplot(1, 4, 3)
    plt.imshow(blur_subtract(image, 30), cmap='gray')
    plt.axis("off")

    fig.add_subplot(1, 4, 4)
    plt.imshow(blur_subtract(image, 100), cmap='gray')
    plt.axis("off")

    # show the images
    plt.show()

filtered = blur_subtract(image)

kSize = 5
kernel = np.ones((kSize, kSize), np.float32) / (kSize * kSize)
dst = cv2.filter2D(filtered, -1, kernel)

ret, thresh = cv2.threshold(dst, 20, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# noise removal, erosion removes boundary pixels
kernel = np.ones((2, 2), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# find contours in the image and make masks from them
cnts = cv2.findContours(opening.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

i, sum = 0, 0
for c in cnts:
    i += 1
    sum += cv2.contourArea(c)
avg_area = sum / i

# loop over the contours
blank = np.zeros(image.shape[:2], dtype="uint8")
hull = []
for c in cnts:
    # compute the center of the contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # draw the center of the contour on the image
    cv2.circle(image_rgb, (cX, cY), 3, (0, 255, 0), -1)
    cv2.circle(blank, (cX, cY), 3, 100, -1)

    # finding the convex hull of a point set
    hull.append(cv2.convexHull(c, False))

cv2.imshow("Contours 1", image_rgb)
cv2.imshow("Contours 2", blank)
cv2.waitKey(0)


"""
# create an empty black image
drawing = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

# draw contours and hull points
color = 255
cv2.drawContours(drawing, hull, -1, color, 1, 8)

cv2.imshow("Hull", drawing)
cv2.waitKey(0)
"""