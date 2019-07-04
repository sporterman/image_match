#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import cv2
import sys
import numpy as np

def get_feature(img):
    '''
    compute color features of bgr image

    :param img:
    :return: feature([3, w*h])
    '''
    #img = cv2.imread(img_path)
    h, w = img.shape[:2]
    chans = cv2.split(img)
    fea = np.zeros((3, h * w))
    for i in range(len(chans)):
        chan = np.array(chans[i]).reshape(1, -1)
        fea[i, :] = chan
    return fea

def compute(fea1, fea2):
    '''
    compute average distance of two vectors

    :param fea1([3, w*h]):
    :param fea2([3, w*h]):
    :return: average distance
    '''
    pixels = 3*fea1.shape[0]*fea1.shape[1]
    dist = np.square(fea1 - fea2)
    dist = np.sum(np.sum(dist, axis=1))
    dist = np.sqrt(dist)
    return dist/pixels

def nms(dets, thresh):
    '''
    nms

    :param dets:
    :param thresh:
    :return: remain indexs, example: [1, ,3 , 6, 7]
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:

        i = order[0]
        keep.append(i)
        #计算窗口i与其他所以窗口的交叠部分的面积，矩阵计算
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #ind为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]
        #下一次计算前要把窗口i去除，所有i对应的在order里的位置是0，所以剩下的加1
        order = order[inds + 1]
    return keep

def get_bbox(img_gray1, img_gray2, w, h, threshold_score):
    '''
    get bounding box through cv2.matchTemplate

    :param img_gray1:
    :param img_gray2:
    :param w:
    :param h:
    :param threshold_score:
    :return:
    '''
    res = cv2.matchTemplate(img_gray1, img_gray2, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold_score)
    bboxes = np.zeros((len(loc[0]), 5))
    for i, pt in enumerate(zip(*loc[::-1])):
        left = int(pt[0])
        top = (pt[1])
        score = res[top, left]
        bboxes[i, :] = np.array([left, top, left + w, top + h, score])
    return bboxes


def main():
    assert len(sys.argv) == 3, "input error"

    threshold_score = 0.9        # bounding box threshold
    threshold_nms = 0.8           # nms threshold
    threshold_dist = 0.4         # distance threshold


    img_rgb = cv2.imread(sys.argv[1])
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    template = cv2.imread(sys.argv[2])
    template_gray = cv2.imread('tanker_a.png', 0)
    h, w = template.shape[:2]

    # get bounding box through cv2.matchTemplate
    bboxes = get_bbox(img_gray, template_gray, w, h, threshold_score)

    # do nms
    idxs = nms(bboxes, threshold_nms)
    bboxes = bboxes[idxs]

    # filter by distance
    dets = []
    fea1 = get_feature(template)
    for box in bboxes:
        fea2 = get_feature(img_rgb[int(box[1]):int(box[3]), int(box[0]):int(box[2])])
        dist = compute(fea1, fea2)
        if dist < threshold_dist:
            dets.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
    for det in dets:
        cv2.rectangle(img_rgb, (det[0], det[1]), (det[2], det[3]), (0, 255, 0), 2)
    cv2.imwrite('res_' + sys.argv[2], img_rgb)
    result = []
    for det in dets:
        result.append((det[1], det[0], det[3], det[2]))
    print(result)

if __name__ == '__main__':
    main()


