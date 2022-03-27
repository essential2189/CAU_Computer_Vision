import cv2
import numpy as np
import matplotlib.pyplot as plt

def img_crop_9x9(img, points_src):
    img_crop1 = img[points_src[0][1] - 100:points_src[0][1] + 100, points_src[0][0] - 100:points_src[0][0] + 100]
    img_crop2 = img[points_src[1][1] - 100:points_src[1][1] + 100, points_src[1][0] - 100:points_src[1][0] + 100]
    img_crop3 = img[points_src[2][1] - 100:points_src[2][1] + 100, points_src[2][0] - 100:points_src[2][0] + 100]
    img_crop4 = img[points_src[3][1] - 100:points_src[3][1] + 100, points_src[3][0] - 100:points_src[3][0] + 100]

    plt.subplot(221)
    plt.imshow(img_crop1)
    plt.subplot(222)
    plt.imshow(img_crop2)
    plt.subplot(223)
    plt.imshow(img_crop3)
    plt.subplot(224)
    plt.imshow(img_crop4)
    plt.show()

    return img_crop1, img_crop2, img_crop3, img_crop4


def cal_hist(img1_pre, img2_pre, points_src, points_src2):
    img1_gray = cv2.cvtColor(img1_pre, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_pre, cv2.COLOR_BGR2GRAY)

    img1_crop1, img1_crop2, img1_crop3, img1_crop4 = img_crop_9x9(img1_gray, points_src)
    img2_crop1, img2_crop2, img2_crop3, img2_crop4 = img_crop_9x9(img2_gray, points_src2)

    hist11 = cv2.calcHist([img1_crop1], [0], None, [254], [0, 254])
    hist12 = cv2.calcHist([img1_crop2], [0], None, [254], [0, 254])
    hist13 = cv2.calcHist([img1_crop3], [0], None, [254], [0, 254])
    hist14 = cv2.calcHist([img1_crop4], [0], None, [254], [0, 254])

    plt.subplot(221)
    plt.plot(hist11)
    plt.subplot(222)
    plt.plot(hist12)
    plt.subplot(223)
    plt.plot(hist13)
    plt.subplot(224)
    plt.plot(hist14)
    plt.show()

    hist21 = cv2.calcHist([img2_crop1], [0], None, [254], [0, 254])
    hist22 = cv2.calcHist([img2_crop2], [0], None, [254], [0, 254])
    hist23 = cv2.calcHist([img2_crop3], [0], None, [254], [0, 254])
    hist24 = cv2.calcHist([img2_crop4], [0], None, [254], [0, 254])

    plt.subplot(221)
    plt.plot(hist21)
    plt.subplot(222)
    plt.plot(hist22)
    plt.subplot(223)
    plt.plot(hist23)
    plt.subplot(224)
    plt.plot(hist24)
    plt.show()

    hist1 = [hist11, hist12, hist13, hist14]
    hist2 = [hist21, hist22, hist23, hist24]

    return hist1, hist2


def comapare_hist(hist1, hist2):
    his_dict1 = {}
    his_dict2 = {}
    his_dict3 = {}
    his_dict4 = {}
    sort_dict = []

    for i in range(4):
        for j in range(4):
            d1 = cv2.compareHist(hist1[i], hist2[j], cv2.HISTCMP_CORREL)
            if i == 0:
                his_dict1['{}'.format(j)] = d1
            if i == 1:
                his_dict2['{}'.format(j)] = d1
            if i == 2:
                his_dict3['{}'.format(j)] = d1
            if i == 3:
                his_dict4['{}'.format(j)] = d1

    sorted_dict1 = sorted(his_dict1.items(), key=lambda item: item[1], reverse=True)
    sorted_dict2 = sorted(his_dict2.items(), key=lambda item: item[1], reverse=True)
    sorted_dict3 = sorted(his_dict3.items(), key=lambda item: item[1], reverse=True)
    sorted_dict4 = sorted(his_dict4.items(), key=lambda item: item[1], reverse=True)

    sort_dict.append(sorted_dict1[0])
    sort_dict.append(sorted_dict2[0])
    sort_dict.append(sorted_dict3[0])
    sort_dict.append(sorted_dict4[0])

    return sort_dict