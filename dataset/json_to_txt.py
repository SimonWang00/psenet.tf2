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
# 1).define your function1;
# 2).define your function2;
# 3).define your function3 and so on;
# ==============================================================================
# LINT.IfChange


import os
import json
import glob
import cv2


# tpath = "./20191204_113515.txt"
jpath = "./20191204_113515.json"

def list_dir(path):
    files = glob.glob(os.path.join(path,'*.json'))
    return files

def load_json(jpath):
    '''
    加载json文本
    Parameters
    ----------
    jpath :

    Returns
    -------

    '''
    f = open(jpath, "r", encoding="utf-8")
    return f.read()

def load_img(pname):
    '''
    read pic
    Parameters
    ----------
    pname :

    Returns
    -------

    '''
    IMG = cv2.imread(pname)
    return IMG

def resize_img(img, resize_w, resize_h):
    '''
    resize pic
    Parameters
    ----------
    img :
    resize_w :
    resize_h :

    Returns
    -------

    '''
    img_resize = cv2.resize(img, (resize_w, resize_h))
    return img_resize

def wapper_annoation(points, label, scalex=1., scaley=1.):
    '''
    组装结果
    Parameters
    ----------
    points : 8个点
    label : label name defult 1
    scalex: default 1
    scaley: default 1

    Returns
    -------

    '''
    annoation = ""
    for x in points:
        annoation = annoation + str(x[0]*scalex) + "," + str(x[1]*scaley) + ","
    annoation = annoation + label
    return annoation

def resize_image(w, h) -> (str, str):
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
    MAX_LEN = 1280
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

def compute_ratio(w, h, resize_w, resize_h):
    '''

    Parameters
    ----------
    w :
    h :
    resize_w :
    resize_h :

    Returns
    -------

    '''
    scalex = resize_w / w
    scaley = resize_h / h
    return scalex, scaley

def record_txt(tpath, annoation):
    '''
    写入转换结果
    Parameters
    ----------
    tpath :
    annoation :

    Returns
    -------

    '''
    f = open(tpath, "a+", encoding="utf-8")
    f.write(annoation + "\n")

def json_to_txt(jpath, scalex, scaley):
    tpath = jpath.replace(".json", ".txt")
    annoation_str = load_json(jpath)
    annoation_dict = json.loads(annoation_str)
    shapes_arr = annoation_dict.get("shapes")
    for shape in shapes_arr:
        points = shape["points"]
        label = shape["label"]
        annoation = wapper_annoation(points, label, scalex=scalex, scaley=scaley)
        record_txt(tpath=tpath, annoation=annoation)

def batch_json_to_txt(path):
    '''
    批量转换
    Parameters
    ----------
    path :

    Returns
    -------

    '''
    files = list_dir(path)
    for f in files:
        pname = f.replace(".json", ".jpg")
        img = load_img(pname)
        h, w = img.shape[0:2]
        resize_w, resize_h = resize_image(w, h)
        scalex, scaley = compute_ratio(w,h, resize_w, resize_h)
        print("resize_w:{} , resize_h:{}".format(resize_w, resize_h))
        print("translate {} to txt".format(f))
        json_to_txt(f, scalex, scaley)
        img = resize_img(img, resize_w, resize_h)
        cv2.imwrite(pname, img)

if __name__ == "__main__":
    path = "D:\\tmp\\118_20200119"
    batch_json_to_txt(path)