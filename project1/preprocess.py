import numpy as np
import cv2
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt

def opening(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    result = cv2.erode(image, kernel2, iterations=4)

    result = cv2.dilate(result, kernel2, iterations=4)

    result = cv2.erode(result, kernel, iterations=2)

    result = cv2.dilate(result, kernel, iterations=2)

    return result


def process(result):
    # low_pass_filter = np.ones((3, 3), np.float32) / 9.0
    # result = cv2.filter2D(result, -1, low_pass_filter)

    # result1 = cv2.fastNlMeansDenoising(result, None, 10, 7, 21)
    result1 = opening(result)

    # result1 = cv2.GaussianBlur(result1, (0, 0), 3)
    # result1 = cv2.medianBlur(result1, 5)
    result1 = cv2.bilateralFilter(result1, 5, 75, 75)


    return result1


def preprocess(img1):
    # img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

    img1_hsv = cv2.GaussianBlur(img1_hsv, (0, 0), 3)
    dst1 = cv2.inRange(img1_hsv, (22, 0, 0), (255, 255, 255))
    plt.subplot(121)
    plt.imshow(dst1)

    result = process(dst1)
    plt.subplot(122)
    plt.imshow(result)
    plt.show()

    bit_wise = cv2.bitwise_and(img1, img1, mask=result)
    bit_wise[bit_wise == 0] = 255

    return bit_wise