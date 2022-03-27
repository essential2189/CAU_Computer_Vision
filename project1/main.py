import cv2
import numpy as np
from utils import get_four_points
from preprocess import preprocess
from cal_hist import cal_hist, comapare_hist

import matplotlib.pyplot as plt


def main():
    img1 = cv2.imread('/Users/essential2189/Desktop/cv_assignment/project1/data/1st.jpg')
    img2 = cv2.imread('/Users/essential2189/Desktop/cv_assignment/project1/data/2nd.jpg')

    print(img1.shape, img2.shape)
    # img1_pre = preprocess(img1)
    # img2_pre = preprocess(img2)
    img1_pre = img1
    img2_pre = img2

    cv2.imshow('Image', img1_pre)
    points_src = get_four_points(img1_pre)

    cv2.imshow('Image', img2_pre)
    points_src2 = get_four_points(img2_pre)

    print(points_src)
    print(points_src2)

    hist1, hist2 = cal_hist(img1_pre, img2_pre, points_src, points_src2)
    sort_dict = comapare_hist(hist1, hist2)

    print(sort_dict)

    raw, col = img1.shape[:2]
    hstack_img = np.hstack([img1, img2])
    hstack_img = cv2.cvtColor(hstack_img, cv2.COLOR_BGR2RGB)
    plt.imshow(hstack_img)

    for l in range(4):
        if sort_dict[l][0] == '0':
            print(0)
            plt.plot([points_src[l][0], points_src2[0][0]+col], [points_src[l][1], points_src2[0][1]])
        elif sort_dict[l][0] == '1':
            print(1)
            plt.plot([points_src[l][0], points_src2[1][0]+col], [points_src[l][1], points_src2[1][1]])
        elif sort_dict[l][0] == '2':
            print(2)
            plt.plot([points_src[l][0], points_src2[2][0]+col], [points_src[l][1], points_src2[2][1]])
        elif sort_dict[l][0] == '3':
            print(3)
            plt.plot([points_src[l][0], points_src2[3][0]+col], [points_src[l][1], points_src2[3][1]])

    plt.show()
    # plt.savefig('/Users/essential2189/Desktop/cv_assignment/project1/result/result.jpg')

if __name__ == '__main__':
    main()