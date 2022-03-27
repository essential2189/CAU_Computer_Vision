import cv2
import numpy as np

def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(data['im'], (x, y), 30, (0,0,255), -1)
        cv2.imshow('Image', data['im'])

        if len(data['points']) < 4:
            data['points'].append([x, y])

def get_four_points(im):
    data = {}
    data['im'] = im.copy()
    data['points'] = []

    cv2.imshow('Image', im)
    cv2.setMouseCallback("Image", mouse_handler,data)
    cv2.waitKey(0)

    points = np.array(data['points'], dtype=int)

    return points


