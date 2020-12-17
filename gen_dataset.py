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

import cv2
import pyclipper 
import os
import glob
import shutil
import numpy as np
import multiprocessing as mp
from itertools import repeat

from utils import del_allfile , convert_label_to_id
from settings import SN, M, label_to_id, DATA_TXT_DIR, DATA_IMG_DIR, TRAINING_DATA_PATH, VALIDATE_DATA_PATH

def read_txt(file):
    '''
    load box
    Parameters
    ----------
    file :

    Returns
    -------

    '''
    with open(file,'r',encoding='utf-8') as f :
        lines = f.read().encode('utf-8').decode('utf-8-sig')
    lines = lines.split('\n')
    gtbox =[]
    for line in lines:
        if(line==''):
            continue
        pts = line.split(',')[0:8]
        if len(pts) >=8:
            # convert str to int
            x1 = round(float(pts[0]))
            y1 = round(float(pts[1]))
            x2 = round(float(pts[2]))
            y2 = round(float(pts[3]))
            x3 = round(float(pts[4]))
            y3 = round(float(pts[5]))
            x4 = round(float(pts[6]))
            y4 = round(float(pts[7]))
        elif len(pts) < 8 and len(pts) >=4:
            # convert str to int
            x1 = round(float(pts[0]))
            y1 = round(float(pts[1]))
            x2 = round(float(pts[2]))
            y2 = round(float(pts[1]))
            x3 = round(float(pts[2]))
            y3 = round(float(pts[3]))
            x4 = round(float(pts[0]))
            y4 = round(float(pts[3]))
        gtbox.append([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    return gtbox

def read_dataset():
    '''
    load dataset
    Returns
    -------

    '''
    files = glob.glob(os.path.join(DATA_TXT_DIR,'*.txt'))
    dataset={}
    for file in files:
        basename = '.'.join(os.path.basename(file).split('.')[:-1])
        imgname = os.path.join(DATA_IMG_DIR,basename+'.jpg')
        gtbox = read_txt(file)
        dataset[imgname] = gtbox
    return dataset

def cal_di(pnt,m,n):
    '''
    calculate di pixels for shrink the original polygon pnt 
    Arg:
        pnt : the points of polygon [[x1,y1],[x2,y2],...]
        m : the minimal scale ration , which the value is (0,1]
        n : the number of kernel scales
    return di_n [di1,di2,...din] 
    '''
    # 计算轮廓面积
    area = cv2.contourArea(pnt)
    # 计算轮廓周长
    perimeter = cv2.arcLength(pnt,True)

    ri_n = [] 
    for i in range(1,n):
        ri = 1.0 - (1.0 - m) * (n - i) / (n-1)
        ri_n.append(ri)

    di_n = []
    for ri in ri_n:
        di = area * (1 - ri * ri ) / perimeter
        di_n.append(di)

    return di_n

 
def shrink_polygon(pnt,di_n):
    '''
    多边形裁剪
    Parameters
    ----------
    pnt :
    di_n :

    Returns
    -------

    '''
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(pnt, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

    shrink_pnt_n = [] 
    for di in di_n:
        shrink_pnt = pco.Execute(-int(di))
        shrink_pnt_n.append(shrink_pnt)
    return shrink_pnt_n


def gen_dataset(data):
    imgname,gtboxes = data[0]
    dst_dir = data[1]
    try:
        basename = '.'.join(os.path.basename(imgname).split('.')[:-1])
        img = cv2.imread(imgname)
        labels = np.ones((SN,img.shape[0],img.shape[1],3))
        labels = labels * 255
        npys = np.zeros((img.shape[0],img.shape[1], SN))

        gtboxes = np.array(gtboxes)
        # shrink 1.0
        for gtbox in gtboxes:
            cv2.drawContours(labels[SN-1],[gtbox],-1,(0,0,255),-1)

        # shrink n-1 times
        for gtbox in gtboxes:
            di_n = cal_di(gtbox, M, SN)
            shrink_pnt_n = shrink_polygon(gtbox,di_n)
            for id,shirnk_pnt in enumerate(shrink_pnt_n):             
                cv2.drawContours(labels[id],np.array(shirnk_pnt),-1,(0,0,255),-1)

        cv2.imwrite(os.path.join(dst_dir,basename+'.jpg'),img)

        # convert labelimage to id
        for idx,label in enumerate(labels):
            npy = convert_label_to_id(label_to_id, label)
            npys[:,:,idx] = npy
        np.save(os.path.join(dst_dir,basename+'.npy'),npys)
    except Exception as e:
        print("Gendataset ERR:%s, imgname: %s"%(e,imgname))
        ddd = './examples/bad'
        try:
            shutil.copyfile(imgname,os.path.join(ddd,basename))
        except:
            pass

def create_dataset():
    data = read_dataset()

    #split trian and test data
    train_num = int(len(data) * 0.8)
    train_data = {key:data[key] for i,key in enumerate(data) if i<train_num }
    test_data  = {key:data[key] for i,key in enumerate(data) if i>=train_num }

    del_allfile(TRAINING_DATA_PATH)
    gen_dataset(train_data)

    del_allfile(VALIDATE_DATA_PATH)
    gen_dataset(test_data)

if __name__=='__main__':
    # create_dataset()
    data = read_dataset()

    # split trian and test data
    train_num = int(len(data) * 0.8)
    train_data = {key:data[key] for i,key in enumerate(data) if i<train_num }
    test_data  = {key:data[key] for i,key in enumerate(data) if i>=train_num }

    del_allfile(TRAINING_DATA_PATH)
    del_allfile(VALIDATE_DATA_PATH)

    with mp.Pool(processes=1) as pool:
        pool.map(gen_dataset,zip(train_data.items(),repeat(TRAINING_DATA_PATH)))
        pool.map(gen_dataset,zip(test_data.items(),repeat(VALIDATE_DATA_PATH)))