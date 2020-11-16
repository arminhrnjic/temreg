from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils
import time
import statistics

start_time = time.time()

# the MSE increases as the images are less similar,
# as opposed to the SSIM where smaller values indicate less similarity;
# the 'Mean Squared Error' between the two images is the
# sum of the squared difference between the two images;
# NOTE: the two images must have the same dimension

checkBlur = False
translateRotate = True


def mse(image1, image2):  # compute the mean squared error
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image2.shape[1])
    return err


def compute_metrics(image1, image2):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(image1, image2)
    s = ssim(image1, image2)

    return m, s


def local_fft(array, ham_filter=True):
    h_f = np.ones_like(array)

    if ham_filter:
        size = array.shape
        x = np.arange(0, size[0])
        y = np.arange(0, size[1])
        xx, yy = np.meshgrid(x, y, sparse=True)
        h_f = (np.sin(xx/(size[0]-1)*np.pi)*np.sin(yy/(size[1]-1)*np.pi))**2

    spectrum = np.fft.fft2(array*h_f, s=None, axes=(-2, -1), norm=None)
    spectrum = np.fft.fftshift(spectrum, axes=(-1, -2))
    spectrum = 20 * np.log(np.abs(spectrum))
    return spectrum


def blur_subtract(array, kernel_size=20):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size*kernel_size)
    dst = cv2.filter2D(array, -1, kernel)

    diff = cv2.subtract(array, dst)
    return diff


def compare_images(image1, image2, increment):
    print("[INFO] comparing images")
    m_min = 10000
    s_max = -1
    best_angle_m = 0
    best_angle_s = 0

    for i in range(-150, 150, 1):
        rotated = imutils.rotate(image1, angle=increment * i)
        m, s = compute_metrics(rotated, image2)
        if m <= m_min:
            m_min = m
            best_angle_m = increment * i
            # print(str(best_angle) + " degrees rotated, new best MSE: " + str(m_min))
        if s >= s_max:
            s_max = s
            best_angle_s = increment * i
            # print(str(best_angle) + " degrees rotated, new best SSIM: " + str(s_max))

    print("Best angle for MSE, lowest MSE, best angle for SSIM, highest SSIM: ")
    print(best_angle_m, m_min, best_angle_s, s_max)
    print("--- %s seconds ---" % (time.time() - start_time))
    return best_angle_m, m_min, best_angle_s, s_max


def compare_fft(image1, image2, increment):
    # for working with FFT, SSIM is much more reliable than MSE for some reason
    print("[INFO] comparing FFT")
    s_max = -1
    best_angle = 0

    for i in range(-150, 150, 1):
        rotated_fft = imutils.rotate(local_fft(image1), angle=increment * i)
        m, s = compute_metrics(rotated_fft, local_fft(image2))

        if s >= s_max:
            s_max = s
            best_angle = increment * i
            # print(str(best_angle) + " degrees rotated, new best SSIM: " + str(s_max))

    print("Best angle, highest SSIM: ")
    print(best_angle, s_max)
    print("--- %s seconds ---" % (time.time() - start_time))

    return best_angle, s_max


def compare_subtracted(image1, image2, increment):
    print("[INFO] comparing blurred and subtracted images")
    m_min = 10000
    s_max = -1
    best_angle_m = 0
    best_angle_s = 0

    for i in range(-150, 150, 1):
        rotated_blur = imutils.rotate(blur_subtract(image1), angle=increment * i)
        m, s = compute_metrics(rotated_blur, blur_subtract(image2))
        if m <= m_min:
            m_min = m
            best_angle_m = increment * i
            # print(str(best_angle) + " degrees rotated, new best MSE: " + str(m_min))
        if s >= s_max:
            s_max = s
            best_angle_s = increment * i
            # print(str(best_angle) + " degrees rotated, new best SSIM: " + str(s_max))

    print("Best angle for MSE, lowest MSE, best angle for SSIM, highest SSIM: ")
    print(best_angle_m, m_min, best_angle_s, s_max)
    print("--- %s seconds ---" % (time.time() - start_time))
    return best_angle_m, m_min, best_angle_s, s_max


original = cv2.imread("S-0224_0.png", 0)
shopped = cv2.imread("Sa-0047_0.png", 0)

if checkBlur:
    # setup the figure
    fig = plt.figure("Original vs. blurred")

    # show first image
    fig.add_subplot(1, 4, 1)
    plt.imshow(original, cmap='gray')
    plt.axis("off")

    # show the second image
    fig.add_subplot(1, 4, 2)
    plt.imshow(blur_subtract(original, 10), cmap='gray')
    plt.axis("off")

    fig.add_subplot(1, 4, 3)
    plt.imshow(blur_subtract(original, 30), cmap='gray')
    plt.axis("off")

    fig.add_subplot(1, 4, 4)
    plt.imshow(blur_subtract(original, 100), cmap='gray')
    plt.axis("off")

    # show the images
    plt.show()


if translateRotate:
    # TRANSLATION
    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(original, 127, 255, 0)
    ret2, thresh2 = cv2.threshold(shopped, 127, 255, 0)

    # calculate moments of binary image
    M = cv2.moments(thresh)
    M2 = cv2.moments(thresh2)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cX2 = int(M2["m10"] / M2["m00"])
    cY2 = int(M2["m01"] / M2["m00"])

    print("Difference in centers of gravity in pixels (x, y): ")
    print(cX - cX2, cY - cY2)

    # We use warpAffine to transform
    # the image using the matrix, T
    T = np.float32([[1, 0, cX - cX2], [0, 1, cY - cY2]])

    height, width = shopped.shape[:2]
    shopped = cv2.warpAffine(shopped, T, (width, height))

    # ROTATION: see functions above
    b1, m1, b2, s1 = compare_images(original, shopped, 0.1)
    b3, s2 = compare_fft(original, shopped, 0.1)
    b4, m3, b5, s3 = compare_subtracted(original, shopped, 0.1)

    angle = statistics.mode([b1, b2, b3, b4, b5])
    print(angle)

    rotated = imutils.rotate(original, angle=angle)
    difference = cv2.subtract(rotated, shopped) + cv2.subtract(shopped, rotated)

    cv2.imshow("Rotated original", rotated)
    cv2.imshow("Translated shopped", shopped)
    cv2.imshow("Difference", difference)
    cv2.waitKey(0)
