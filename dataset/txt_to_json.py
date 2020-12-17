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

"""translate txt to json package"""

import os
import json
import glob
import base64
from PIL import Image

tpath = "./20191204_113515.txt"
jpath = "./20191204_113515.json"


def load_file(path):
    '''
    load file from disk
    Parameters
    ----------
    path :

    Returns
    -------

    '''
    with open(path, "r+", encoding="utf-8") as f:
        content = f.read()
        return content

def mark_format(zb_arr):
    '''
    在此处填入坐标
    Parameters
    ----------
    zb_arr : 坐标列表

    Returns shape_format
    -------

    '''
    shapes = []
    if isinstance(zb_arr, list):
        for zb in zb_arr:
            shape_format = {
                            "label": "1",
                            "line_color": None,
                            "fill_color": None,
                            "points": zb,
                            "shape_type": "polygon",
                            "flags": {}
                        }
            shapes.append(shape_format)
    return shapes


def labelme_format(shape_arr,imagePath,imageData ,shape):
    '''
    package json data
    Parameters
    ----------
    shape_arr : list
    imagePath : "./20191204_113515.jpg"
    imageData :picture  base64 encode
    shape :picture shape

    Returns
    -------

    '''
    json_format = {
                  "version": "3.16.7",
                  "flags": {},
                  "shapes": shape_arr,
                  "lineColor": [
                            0,
                            255,
                            0,
                            128
                          ],
                  "fillColor": [
                            255,
                            0,
                            0,
                            128
                          ],
                "imagePath": imagePath,
                "imageData": imageData,
                "imageHeight": shape[1],
                "imageWidth": shape[0]
                }
    return json_format

def exract_imagePath(path):
    '''
    提取图片名称
    Parameters
    ----------
    path :

    Returns
    -------

    '''
    if ".txt" in path:
        imagePath = path.replace(".txt", ".jpg")
        return imagePath
    raise Exception("only support txt file . base on exract_imagePath(path)")

def exract_jsonPath(path):
    '''
    提取应该生成的json文件名称
    Parameters
    ----------
    path : txt的名称

    Returns
    -------

    '''
    if ".txt" in path:
        jsonPath = path.replace(".txt", ".json")
        return jsonPath
    raise Exception("only support txt file . base on exract_jsonPath(path)")

def extract_shape(path):
    '''
    提取图片的形状
    Parameters
    ----------
    path :

    Returns
    -------

    '''
    img = Image.open(path)
    shape =img.size
    return shape

def extract_zb(path):
    '''
    提取坐标，只读8个点的
    Parameters
    ----------
    path :

    Returns zb_arr
    -------

    '''
    content = load_file(path)
    content_arr = content.split("\n")   # 转换为list
    zb_arr = []
    for zb in content_arr:
        if "," in zb:
            zb = zb.split(",")
            if len(zb) >= 8:
                zb = zb[0:8]
                zb_arr.append([[float(zb[0]), float(zb[1])], [float(zb[2]), float(zb[3])], [float(zb[4]), float(zb[5])], \
                               [float(zb[6]), float(zb[7])]])
            elif len(zb) >=4 and len(zb) < 8:
                zb = zb[0:4]
                x1 = round(float(zb[0]))
                y1 = round(float(zb[1]))
                x2 = round(float(zb[2]))
                y2 = round(float(zb[1]))
                x3 = round(float(zb[2]))
                y3 = round(float(zb[3]))
                x4 = round(float(zb[0]))
                y4 = round(float(zb[3]))
                zb_arr.append([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    if zb_arr:
        return zb_arr

def load_imageData(imagePath):
    '''
    对图片进行base64转码
    Parameters
    ----------
    imagePath :

    Returns
    -------

    '''
    pic = open(imagePath,"rb").read()
    imageData = base64.b64encode(pic)
    imageData = str(imageData,"utf-8")
    return imageData

def record_json_format(json_format, path):
    '''
    形成json文件
    Parameters
    ----------
    json_format :
    path :

    Returns
    -------

    '''
    with open(path, "w", encoding="utf-8") as f:
        json_format = json.dumps(json_format,ensure_ascii=False,indent=4)
        f.write(json_format)

def load_txt_to_json(path):
    '''
    将坐标转换为labelme 生成的json文件
    Parameters
    ----------
    path :

    Returns None
    -------

    '''
    zb_arr = extract_zb(path)
    shape_format = mark_format(zb_arr)
    imagePath = exract_imagePath(path)
    jsonPath = exract_jsonPath(path)
    imageData = load_imageData(imagePath)
    shape = extract_shape(imagePath)
    json_format = labelme_format(shape_format, imagePath, imageData, shape)
    record_json_format(json_format, jsonPath)

def list_dir(path):
    txt_files = glob.glob(os.path.join(path,'*.txt'))
    return txt_files

def batch_txt_to_json(path):
    '''
    批量转换成json
    Parameters
    ----------
    path :

    Returns
    -------

    '''
    txt_files = list_dir(path)
    for txt in txt_files:
        load_txt_to_json(txt)

if __name__ == "__main__":
    path = "D:\\tmp\\118_20200119"
    batch_txt_to_json(path)