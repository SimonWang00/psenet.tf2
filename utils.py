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

import threading
import numpy as np 
import os
import cv2
import glob
from numba import jit
from scipy.ndimage import filters, interpolation
from numpy import amin, amax


def translate_4_to_8(points_4):
    '''
    定位框4个点转换为8个
    Parameters
    ----------
    points_4 :

    Returns
    -------

    '''
    gtbox = []
    if len(points_4) ==4:
        x1 = round(float(points_4[0]))
        y1 = round(float(points_4[1]))
        x2 = round(float(points_4[2]))
        y2 = round(float(points_4[1]))
        x3 = round(float(points_4[2]))
        y3 = round(float(points_4[3]))
        x4 = round(float(points_4[0]))
        y4 = round(float(points_4[3]))
        gtbox.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    raise Exception("Please check points_4")

class BatchIndices():
    def __init__(self,total,batchsize,trainable=True):
        self.n = total
        self.bs = batchsize
        self.shuffle = trainable
        self.lock = threading.Lock()
        self.reset()
    def reset(self):
        self.index = np.random.permutation(self.n) if self.shuffle==True else np.arange(0,self.n)
        self.curr = 0
    
    def __next__(self):
        with self.lock:
            if self.curr >= self.n:
                self.reset()
            rn = min(self.bs,self.n - self.curr)
            res = self.index[self.curr:self.curr+rn]
            self.curr += rn
            return res

def del_allfile(path):
    '''
    del all files in the specified directory
    '''
    filelist = glob.glob(os.path.join(path,'*.*'))
    for f in filelist:
        os.remove(os.path.join(path,f))

def convert_label_to_id(label2id,labelimg):
    '''
    convert label image to id npy
    param:
    labelimg - a label image with 3 channels
    label2id  - dict eg.{(0,0,0):0,(0,255,0):1,....}
    '''

    h,w = labelimg.shape[0],labelimg.shape[1]
    npy = np.zeros((h,w),'uint8')
    
    for i,j in label2id.items():
        idx = ((labelimg == i) * 1)
        idx = np.sum(idx, axis=2) >=3
        npy = npy + idx * j

    return npy


def convert_id_to_label(id,label2id):
    '''
    convet id numpy to label image 
    param:
    id          : numpy
    label2id  - dict eg.{(0,0,0):0,(0,255,0):1,....}
    return labelimage 
    '''
    h,w = id.shape[0],id.shape[1]

    labelimage = np.ones((h,w,3),'uint8') * 255
    for i,j in label2id.items():
        labelimage[np.where(id==j)] = i 

    return labelimage
 

@jit
def ufunc_4(S1,S2,TAG):
    '''
    #indices 四邻域 x-1 x+1 y-1 y+1，如果等于TAG 则赋值为label
    Parameters
    ----------
    S1 :
    S2 :
    TAG :

    Returns
    -------

    '''
    for h in range(1,S1.shape[0]-1):
        for w in range(1,S1.shape[1]-1):
            label = S1[h][w]
            if(label!=0):
                if(S2[h][w-1] == TAG):                          
                    S2[h][w-1] = label
                if(S2[h][w+1] == TAG):                            
                    S2[h][w+1] = label
                if(S2[h-1][w] == TAG):                            
                    S2[h-1][w] = label
                if(S2[h+1][w] == TAG):                           
                    S2[h+1][w] = label
                    
def scale_expand_kernel(S1,S2):
    TAG = 10240                     
    S2[S2==255] = TAG
    mask = (S1!=0)
    S2[mask] = S1[mask]
    cond = True 
    while(cond):  
        before = np.count_nonzero(S1==0)
        ufunc_4(S1,S2,TAG)  
        S1[S2!=TAG] = S2[S2!=TAG]  
        after = np.count_nonzero(S1==0)
        if(before<=after):
            cond = False
    return S1

def filter_label_by_area(labelimge,num_label,area=5):
    for i in range(1,num_label+1):
        if(np.count_nonzero(labelimge==i)<=area):
            labelimge[labelimge==i] ==0
    return labelimge

def scale_expand_kernels(kernels,filter=False):
    '''
    args:
        kernels : S(0,1,2,..n) scale kernels , Sn is the largest kernel
    '''
    S = kernels[0]
    # paper中的CC操作
    num_label,labelimage = cv2.connectedComponents(S.astype('uint8'))
    if(filter==True):
        labelimage = filter_label_by_area(labelimage,num_label)
    for Si in kernels[1:]:
        labelimage = scale_expand_kernel(labelimage,Si)
    return num_label,labelimage   

def fit_minarearectange(num_label,labelImage):
    rects= []
    for label in range(1,num_label+1):
        points = np.array(np.where(labelImage == label)[::-1]).T

        rect = cv2.minAreaRect(points)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        area = cv2.contourArea(rect)
        if(area<10):
            # print('area:',area)
            continue
        rects.append(rect)
    return rects

@jit(nopython=True)
def fit_minarearectange_2(num_label,labelImage) -> 6:
    '''
    最小外接矩形优化
    '''
    points = [[]] * num_label
    for h in range(0,labelImage.shape[0]):
        for w in range(0,labelImage.shape[1]):
            value = labelImage[h][w]
            if(value > 0):
                points[value-1].append([w,h]) 
    return 6

def save_MTWI_2108_resault(filename,rects,scalex=1.0,scaley=1.0):
    with open(filename,'w',encoding='utf-8') as f:
        for rect in rects:
            line = ''
            for r in rect:
                line += str(r[0] * scalex) + ',' + str(r[1] * scaley) + ','
            line = line[:-1] + '\n'
            f.writelines(line)

def fit_boundingRect(num_label,labelImage) -> list:
    rects= []
    for label in range(1,num_label+1):
        points = np.array(np.where(labelImage == label)[::-1]).T
        #rect = cv2.minAreaRect(points)
        #rect = cv2.boxPoints(rect)
        #rect = np.int0(rect)
        x,y,w,h = cv2.boundingRect(points)
        rect = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
        rects.append(rect)
    return rects

def fit_boundingRect_2(num_label,labelImage) -> list:
    rects= []
    for label in range(1,num_label+1):
        points = np.array(np.where(labelImage == label)[::-1]).T
        x,y,w,h = cv2.boundingRect(points)
        rect = np.array([x,y,x+w,y+h])
        rects.append(rect)
    return rects


def rotate_cut_img(im, degree, box, w, h, leftAdjust=True, rightAdjust=True, alph=0.2) -> tuple:
    '''
    剪切图片
    Parameters
    ----------
    im :
    degree : 角度
    box : 定位框
    w :
    h :
    leftAdjust : 左边是否需要调整
    rightAdjust : 右边是否需要调整
    alph : 调整比例

    Returns
    -------

    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    x_center, y_center = np.mean([x1, x2, x3, x4]), np.mean([y1, y2, y3, y4])
    degree_ = degree * 180.0 / np.pi
    right = 0
    left = 0
    if rightAdjust:
        right = 1
    if leftAdjust:
        left = 1

    box = (max(1, x_center - w / 2 - left * alph * (w / 2))  ##xmin
           , y_center - h / 2,  ##ymin
           min(x_center + w / 2 + right * alph * (w / 2), im.size[0] - 1)  ##xmax
           , y_center + h / 2)  ##ymax

    newW = box[2] - box[0]
    newH = box[3] - box[1]
    tmpImg = im.rotate(degree_, center=(x_center, y_center)).crop(box)
    return tmpImg, newW, newH


def estimate_skew_angle(raw):
    """
    估计图像文字角度
    """

    def resize_im (im, scale, max_scale=None ):
        f = float(scale) / min(im.shape[0], im.shape[1])
        if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
            f = float(max_scale) / max(im.shape[0], im.shape[1])
        return cv2.resize(im, (0, 0), fx=f, fy=f)

    raw = resize_im(raw, scale=600, max_scale=900)
    image = raw - amin(raw)
    image = image / amax(image)
    m = interpolation.zoom(image, 0.5)
    m = filters.percentile_filter(m, 80, size=(20, 2))
    m = filters.percentile_filter(m, 80, size=(2, 20))
    m = interpolation.zoom(m, 1.0 / 0.5)

    w, h = min(image.shape[1], m.shape[1]), min(image.shape[0], m.shape[0])
    flat = np.clip(image[:h, :w] - m[:h, :w] + 1, 0, 1)
    d0, d1 = flat.shape
    o0, o1 = int(0.1 * d0), int(0.1 * d1)
    flat = amax(flat) - flat
    flat -= amin(flat)
    est = flat[o0:d0 - o0, o1:d1 - o1]
    angles = range(-15, 15)
    estimates = []
    for a in angles:
        roest = interpolation.rotate(est, a, order=0, mode='constant')
        v = np.mean(roest, axis=1)
        v = np.var(v)
        estimates.append((v, a))

    _, a = max(estimates)
    return a


class text_porposcal:
    '''
    定位框合并
    '''
    def __init__(self, rects,imgw, max_dist =50, threshold_overlap_v = 0.5):
        self.rects = np.array(rects) 
        self.imgw = imgw
        self.max_dist = max_dist 
        self.threshold_overlap_v = threshold_overlap_v
        self.graph = np.zeros((self.rects.shape[0],self.rects.shape[0]))
        self.r_index = [[] for _ in range(imgw)]
        for index , rect in enumerate(rects):
            self.r_index[int(rect[0])].append(index)

    def get_sucession(self,index):
        rect = self.rects[index]
        for left in range(rect[0]+1,min(self.imgw-1,rect[2]+self.max_dist)):
            for idx in self.r_index[left]:
                if(self.meet_v_iou(index,idx) > self.threshold_overlap_v):
                    return idx 
        return -1

    def meet_v_iou(self,index1,index2):
        height1 = self.rects[index1][3] - self.rects[index1][1]
        height2 = self.rects[index2][3] - self.rects[index2][1]
        y0 = max(self.rects[index1][1],self.rects[index2][1])
        y1 = min(self.rects[index1][3],self.rects[index2][3])
        
        overlap_v = max(0,y1- y0)/max(height1,height2)
        return overlap_v

    def sub_graphs_connected(self):
        sub_graphs=[]
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v=index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v=np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs

    def fit_line(self,text_boxes):
        '''
        先用所有text_boxes的最大外包点做，后期可以用线拟合试试
        '''
        x1 = np.min(text_boxes[:,0])
        y1 = np.min(text_boxes[:,1])
        x2 = np.max(text_boxes[:,2])
        y2 = np.max(text_boxes[:,3])
        return [x1,y1,x2,y2]

    def get_text_line(self) -> np.array:
        for idx ,_ in enumerate(self.rects):
            sucession = self.get_sucession(idx)
            if(sucession>0):
                self.graph[idx][sucession] = 1 
                
        sub_graphs = self.sub_graphs_connected()

        # 独立未合并的框
        set_element = set([y for x in sub_graphs for y in x])
        for idx,_ in enumerate(self.rects):
            if(idx not in set_element):
                sub_graphs.append([idx])
        
        text_boxes = []
        for sub_graph in sub_graphs:
            tb = self.rects[list(sub_graph)]
            tb = self.fit_line(tb)
            text_boxes.append(tb)

        return np.array(text_boxes)

def seg_box_img(img,rets) -> list:
    '''
    保存定位框图片
    :param img:图片
    :param rets:
    :return:
    '''
    # 用于记录每个定位框对应的坐标
    box_list = []
    # 对定位框进行遍历
    for i,ret in enumerate(rets):
        # startX, startY, endX, endY
        # img:500 * 354 ,eg:[217, 56, 267, 71]
        if 0 not in ret:
            box_img = img[ret[1]:ret[3], ret[0]:ret[2],:]
            path = './{}.png'.format(i)
            cv2.imwrite(path, box_img)
            box_list.append(box_img)
    return box_list

def adjust_side(rects, left_adjust=True, right_adjust=True) -> list:
    '''
    是否左右调整box 边界误差，解决文字漏检

    Parameters
    ----------
    rects : list
    left_adjust : bool
    right_adjust : bool

    Returns
    -------
    Notes:
    # startX, startY, endX, endY
    # img:500 * 354 ,eg:[217, 56, 267, 71]
    TODO:
    先只调整横向左右的距离吧
    '''
    left_rate = 0
    right_rate = 0
    if left_adjust:
        left_rate = -0.05
    if right_adjust:
        right_rate = 0.025
    rects = [[int(rt[0] * (2 + left_rate)), int(rt[1] * 2 -2), int(rt[2] * (2 + right_rate)), int(rt[3] * 2) + 2] for rt in rects]
    return rects