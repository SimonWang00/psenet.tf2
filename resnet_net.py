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
# 1).building 2 layers of resnet block;
# 2).building 3 layers of resnet block;
# 3).define my resnet networks;
# ==============================================================================
# LINT.IfChange

"""define my resnet package"""


import tensorflow as tf
from typing import Tuple, List
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Activation

def identity_block(input_data:tf.Tensor,
                   kernel_size: int or 3,
                   filters: Tuple[int, int, int],
                   stage: str or int,
                   block: str,
                   use_bias=False, train_bn=True) -> tf.Tensor:
    '''
    building 2 layers of resnet block

    The alias's identity_block is the block that has no conv layer at shortcut

    Parameters
    ----------
    input_data :
    kernel_size : default 3, the kernel size of middle conv layer at main path
    filters : list of integers, the nb_filters of 3 conv layer at main path
    stage : str or int, current stage label, used for generating layer names
    block : laber name, 'a','b'..., current block label, used for generating layer names
    use_bias : is use bias
    train_bn : is use BatchNormalization

    Returns
    -------

    '''
    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, (1, 1), padding="SAME", name=conv_name_base + '2a', use_bias=use_bias)(input_data)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filter2, (kernel_size, kernel_size), padding="SAME", name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), padding="SAME", name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

    x = tf.keras.layers.Add()([x, input_data])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_block_3(input_data,
                   kernel_size: int or 3,
                   filters: Tuple[int, int, int],
                   stage: str or int,
                   block: str,
                   strides=(2, 2) or Tuple[int, int],
                   use_bias=False,
                   train_bn=True) -> tf.Tensor:
    '''

    building 3 layers of resnet block
    projection shortcut

    Parameters
    ----------
    input_data :
    kernel_size :default 3, the kernel size of middle conv layer at main path
    filters : list of integers, the nb_filters of 3 conv layer at main path
    stage : current stage label, used for generating layer names
    block : 'a','b'..., current block label, used for generating layer names
    strides : the shortcut should have subsample=(2,2)
    use_bias :
    train_bn :

    Returns

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    -------

    '''

    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, (1, 1), strides=strides, padding="SAME", name=conv_name_base + '2a', use_bias=use_bias)(input_data)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filter2, (kernel_size, kernel_size), padding="SAME", name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), padding="SAME", name=conv_name_base +'2c', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = Conv2D(filter3, (1, 1), padding="SAME", strides=strides, name=conv_name_base + '1', use_bias=use_bias)(input_data)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = tf.keras.layers.Add()([x, shortcut])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def build_resnet(input_image,
                 architecture: str,
                 stage5=False, train_bn=True ) -> List:
    '''
    Build the CRNN network

    Parameters
    ----------
    input_image : inputs
    architecture : Can be resnet50 or resnet101
    stage5 : Boolean, If False, stage5 of the network is not created
    train_bn : Boolean, is use BatchNormalization

    Returns
    -------

    '''
    assert architecture in ["resnet50", "resnet101"]

    # Stage 1: input(b, 300, 1000, 3) -> output(b, 75, 250, 64)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(input_image)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNormalization(name='bn_conv1')(x, training=train_bn)
    x = Activation('relu')(x)
    C1 = x = MaxPooling2D((3, 3), strides=(2, 2), padding="SAME")(x)

    # Stage 2: input(b, 75, 250, 64) -> output(b, 75, 250, 256)
    x = resnet_block_3(x, 3, (64, 64, 256), stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, (64, 64, 256), stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, (64, 64, 256), stage=2, block='c', train_bn=train_bn)

    # Stage 3: input(b, 75, 250, 256) -> output(b, 38, 125, 512)
    x = resnet_block_3(x, 3, (128, 128, 512), stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, (128, 128, 512), stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, (128, 128, 512), stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, (128, 128, 512), stage=3, block='d', train_bn=train_bn)

    # Stage 4: input(b, 38, 125, 512) -> output(b, 19, 63, 1024)
    x = resnet_block_3(x, 3, (256, 256, 1024), stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, (256, 256, 1024), stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x

    # Stage 5: input(b, 19, 63, 1024) -> output(b, 19, 63, 2048)
    if stage5:
        x = resnet_block_3(x, 3, (512, 512, 2048), stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, (512, 512, 2048), stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, (512, 512, 2048), stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]