#!/usr/bin/python3
# __*__ coding: utf-8 __*__

'''
@Author: SimonWang00
@Os：Windows 10 x64
@Contact: simon_wang00@163.com
@Software: PY PyCharm
@File: settings.py
@Time: 2020/12/15 15:15
'''

# Copyright 2020 The SimonWang00. All Rights Reserved.
#
# @Desc:
# 1).restore my checkpoint;
# 2).restore input size;
# 3).text_detector;
# ==============================================================================
# LINT.IfChange

"""Using psenet for text detector package"""

import cv2
import numpy as np
import tensorflow as tf
from model import __model
from utils import scale_expand_kernels, fit_boundingRect_2, text_porposcal, adjust_side

MODEL_PATH = "./models/"
MIN_LEN = 640
MAX_LEN = 1280     # 调大是利于小目标的检测2560

def text_detector_model(path = MODEL_PATH):
    '''
    restore my psenet model
    Returns model object
    -------

    '''
    model = __model()
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory=path, max_to_keep=2)
    checkpoint.restore(manager.latest_checkpoint)
    return model

def resize_image(w, h):
    '''
    resize image to a size multiple of 32 which is required by the network
    '''
    if (w < h):
        if (h < MIN_LEN):
            scale = 1.0 * MIN_LEN / h
            w = w * scale
            h = MIN_LEN
        elif (h > MAX_LEN):
            scale = 1.0 * MAX_LEN / h
            w = w * scale
            h = MAX_LEN
    elif (h <= w):
        if (w < MIN_LEN):
            scale = 1.0 * MIN_LEN / w
            h = scale * h
            w = MIN_LEN
        elif (w > MAX_LEN):
            scale = 1.0 * MAX_LEN / w
            h = scale * w
            h = MAX_LEN

    w = int(w // 32 * 32)
    h = int(h // 32 * 32)

    return w, h

def resize_image2(w, h):
    '''
    改进型
    resize image to a size multiple of 32 which is required by the network
    Parameters
    ----------
    w :
    h :

    Returns
    -------

    '''
    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > MAX_LEN:
        ratio = float(MAX_LEN) / resize_h if resize_h > resize_w else float(MAX_LEN) / resize_w
    else:
        ratio = 1.

    # ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w

    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32

    return resize_w, resize_h

def recover_size(rects, scalex, scaley):
    '''
    将定位框映射到原图中进行适配
    Parameters
    ----------
    rects :
    scalex :
    scaley :

    Returns
    -------

    '''
    # 将定位框长和宽扩大一倍
    rects = [rect * 2 for rect in rects]
    rects_orign = []
    for rect in rects:
        rects_orign.append(
            np.array([round(rect[0] * scalex), round(rect[1] * scaley), round(rect[2] * scalex), round(rect[3] * scaley)]))
    return rects_orign

def show_rects(rects, images, pname):
    '''
    显示定位框
    Parameters
    ----------
    rects : 定位框
    images : 图片
    pname : 保存图片名称

    Returns
    -------

    '''
    for rt in rects:
        cv2.rectangle(images, (rt[0], rt[1]),(rt[2] ,rt[3]), (255, 0, 0), 2)
    cv2.imwrite(pname, images)


def predict(images):
    '''
    detect all the texts of picture.
    Parameters
    ----------
    images :

    Returns
    -------

    '''
    images_orign = images
    h, w = images.shape[0:2]
    w, h = resize_image2(w, h)
    print("resize_w:{} , resize_h:{}".format(w, h))

    scalex = images.shape[1] / w
    scaley = images.shape[0] / h

    images = cv2.resize(images, (w, h), cv2.INTER_AREA)
    # input [b, h, w, channels]
    images = np.reshape(images, (1, h, w, 3))

    # restore my model
    model = text_detector_model()
    res = model.predict(images[0:1, :, :, :])

    res1 = res[0]
    res1[res1 > 0.9] = 1
    res1[res1 <= 0.9] = 0
    newres1 = []
    for i in [2, 4]:
        n = np.logical_and(res1[:, :, 5], res1[:, :, i]) * 255
        newres1.append(n)
    newres1.append(res1[:, :, 5] * 255)

    # 像素扩张算法
    num_label, labelimage = scale_expand_kernels(newres1, filter=False)
    # 精裁剪，最小外接矩阵优化
    rects = fit_boundingRect_2(num_label, labelimage)
    print("定位框个数：", len(rects))
    # new_w = int(w * 0.5)
    # new_h = int(h * 0.5)
    # im = cv2.resize(images[0], (new_w, new_h))
    # 原始定位框
    # show_rects(rects, im, pname="test_4.jpg")

    # 进行文本框合并
    g = text_porposcal(rects, res1.shape[1], max_dist=10, threshold_overlap_v=0.5)
    rects = g.get_text_line()
    rects = recover_size(rects, scalex, scaley)     # 恢复原尺寸

    rects = adjust_side(rects, left_adjust=True, right_adjust=False)
    show_rects(rects, images_orign, pname="test.jpg")
    # estimate_skew_angle()
    # 进行实际切割实验
    # seg_box_img(images_orign, rects)
    return rects

if __name__ == "__main__":
    import time
    t0 = time.time()
    IMG_PATH = "./invoice.jpg"
    IMG = cv2.imread(IMG_PATH)
    rects = predict(IMG)
    print("It cost {} s".format(time.time()-t0))