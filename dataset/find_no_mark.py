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

"""xxxxx package"""

import os
import glob
import shutil

def list_dir(path):
    '''
    找出所有的文件
    Parameters
    ----------
    path :

    Returns
    -------

    '''
    json_files = glob.glob(os.path.join(path,'*.json'))
    jpg_files = glob.glob(os.path.join(path,'*.jpg'))
    return json_files, jpg_files

def find_no_mark(json_files, jpg_files):
    '''
    分别找出标记和没标记的文件
    Parameters
    ----------
    json_files :
    jpg_files :

    Returns
    -------

    '''
    mark_arr_jpg = []
    mark_arr_json = []
    no_mark_arr = []
    for jf in jpg_files:
        _jf = jf.replace(".jpg", ".json")
        if _jf not in json_files:
            no_mark_arr.append(jf)
        else:
            mark_arr_jpg.append(jf)
            mark_arr_json.append(_jf)
    return mark_arr_jpg, mark_arr_json, no_mark_arr

def save_mark_file(fname):
    '''
    复制标记过的文件：jpg和json
    Parameters
    ----------
    fname :

    Returns
    -------

    '''
    mark_path = "./mark"
    if not os.path.exists(mark_path):
        os.mkdir(mark_path)
    shutil.copy(fname, mark_path)

def save_nomark_file(fname):
    '''
    复制没有标记的文件
    Parameters
    ----------
    fname :

    Returns
    -------

    '''
    no_mark_path = "./nomark"
    if not os.path.exists(no_mark_path):
        os.mkdir(no_mark_path)
    shutil.copy(fname, no_mark_path)


def batch_find(path):
    json_files, jpg_files = list_dir(path)
    mark_arr_jpg, mark_arr_json, no_mark_arr = find_no_mark(json_files, jpg_files)
    for mp in mark_arr_jpg:
        save_mark_file(mp)

    for mj in mark_arr_json:
        save_mark_file(mj)

    for nm in no_mark_arr:
        save_nomark_file(nm)

if __name__ == "__main__":
    input_path = input("(#`O′)--> 请输入您labelme的标记目录：")
    batch_find(input_path)
    print("执行完毕！")
    print("程序退出执行")