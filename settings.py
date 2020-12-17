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

"""PseNet Project's config"""

TRAIN_BN = False
IMAGE_MIN_DIM = 800
IMAGE_MAX_DIM = 1024
IMAGE_CHANNEL_COUNT = 3
IMAGE_SHAPE = (IMAGE_MIN_DIM,IMAGE_MAX_DIM,IMAGE_CHANNEL_COUNT)

# FPN Confihs, all outputs should fit this size
TOP_DOWN_PYRAMID_SIZE = 256

# Network Configs
BACKBONE = "resnet50"  # resnet101 or resnet50
LEARNING_RATE = 0.0001
MAX_IMAGE_LARGE_SIDE = 1280
MAX_TEXT_SIZE = 800
MIN_TEXT_AREA_SIZE = 10


# the number of kernel scales
OUTPUTS = 6
# dice loss
BATCH_LOSS = True
# positive ：negtive = 1:rate_ohem
RATE_OHEM = 3
# number of kernel scales
SN = 6
# balances the importance between Lc and Ls
RATE_LC_LS = 0.7
# metric iou
METRIC_IOU_BATCH = True

# tensorboard logs dir
LOGDIR = "./logs"
# 1s, 2s and 4s means the width and height of the output map are 1/1, 1/2 and 1/4 of the input
NS = 4
M = 0.5  #the minimal scale ration , which the value is (0,1]

#随机剪切 文字区域最小面积
data_gen_clip_min_area = 20*100
data_gen_min_scales = 0.8
data_gen_max_scales = 2.0
data_gen_itter_scales = 0.3


# 预处理中的定位框和输入图片
DATA_TXT_DIR = './examples/invoice/txt'
DATA_IMG_DIR = './examples/invoice/image'

# 生成输入数据二进制
TRAINING_DATA_PATH = "./examples/invoice_train_label"
VALIDATE_DATA_PATH = "./examples/invoice_test_label"

# 训练配置参数列表
label_to_id = {(255,255,255):0,(0,0,255):1}
max_to_keep = 2             # 表示保存近2个迭代的生成模型
RESTORE = True              # true-代表恢复模型；false-代表不恢复最近一次模型
N_EPOCH = 500               # 迭代次数
shape = (640, 640)          # 设置输入的图片尺寸
batch_size = 1              # 批处理中输入的图片数目
num_class = 2               # 二分类
MODEL_PATH = "./models/"    # 生成模型的路径