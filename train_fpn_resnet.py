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

"""train psenet implemention package"""
import sys
sys.path.append("..")

import os
import time
import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model
from model import __model, build_loss, build_iou, mean_iou
from callbacks import TensorBoardCallback, CheckpointSaveCallback
from generator import Generator
from settings import LEARNING_RATE, LOGDIR, MODEL_PATH, TRAINING_DATA_PATH, VALIDATE_DATA_PATH, max_to_keep,batch_size,num_class,shape,N_EPOCH

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# GPU的坑
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)

@tf.function
def train_one_step(model, X, Y,optimizer):
    '''
    update weights using auto gradient and adam
    Parameters
    ----------
    model : psenet + fpn
    X : input images
    Y : label
    optimizer : trian optimizer

    Returns
    -------

    '''
    with tf.GradientTape() as tape:
        X = tf.cast(X, tf.float32)
        Y = tf.cast(Y, tf.float32)
        y_pred = model(X, training=True)
        loss = build_loss(y_true=Y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
    return loss

@tf.function
def val_one_step(model, X, Y, total_iou):
    '''
    compute valitation datasets loss and accuracy.

    Parameters
    ----------
    model : psenet + fpn
    X : images
    Y : labels

    Returns loss, accuracy
    -------

    '''
    X = tf.cast(X, tf.float32)
    Y = tf.cast(Y, tf.float32)
    y_pred = model(X)
    loss = build_loss(y_true=Y,y_pred=y_pred)
    loss = tf.reduce_mean(loss)
    accuracy, total_iou = mean_iou(y_true=Y, y_pred=y_pred, total_iou=total_iou)
    return loss, accuracy

def my_optimizer(optimizer_model:str=None):
    '''
    define your optimizer.
    support adam and sgd.
    default adam
    Parameters
    ----------
    optimizer_model :str model choice

    Returns optimizer object
    -------

    '''
    if optimizer_model =="adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    elif optimizer_model =="sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    return optimizer


def train():
    '''
    train my psenet
    Returns
    -------

    '''

    batch_size = 64
    num_class = 2
    shape = (640, 640)

    # add summary
    tf.summary.scalar('learning_rate', LEARNING_RATE)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    # 单个GPU
    psenet_model = __model()
    # # 多GPU
    # psenet_model = multi_gpu_model(psenet_model, 2)
    ious = build_iou([0, 1], ['background', 'text'])
    tb = TensorBoardCallback(LOGDIR)
    checkpoint = CheckpointSaveCallback(MODEL_PATH)
    callbacks_arr = [checkpoint, tb]

    # package my model
    # psenet_model.compile(loss=build_loss, optimizer=optimizer, metrics=[ious,])

    # iter dataset
    data_train = Generator(TRAINING_DATA_PATH, batch_size=batch_size, istraining=True, num_classes=num_class, mirror=False,
                           reshape=shape)
    data_validate = Generator(VALIDATE_DATA_PATH, batch_size=batch_size, istraining=False, num_classes=num_class,
                              reshape=shape, mirror=False, scale=False, clip=False, trans_color=False)

    # fit model
    # history = psenet_model.fit_generator(next(data_train),
    #                                      steps_per_epoch=data_train.num_samples() // batch_size,
    #                                      epochs=100,
    #                                      validation_data=next(data_validate),
    #                                      validation_steps=data_validate.num_samples() // batch_size,
    #                                      verbose=1,
    #                                      initial_epoch=0,
    #                                      workers=4,
    #                                      callbacks=callbacks_arr)

    history = psenet_model.fit(data_train.__next__(),
                               batch_size=8,
                               epochs = 100,
                               validation_data=data_validate.__next__(),
                               validation_steps=data_validate.num_samples()//batch_size,
                               verbose=1,
                               initial_epoch=0,
                               callbacks=callbacks_arr)

if __name__ == "__main__":
    localtime = time.strftime("%Y%m%d%H", time.localtime())

    # add summary
    optimizer = my_optimizer("adam")
    # 单个GPU
    psenet_model = __model()
    # # 多GPU
    # psenet_model = multi_gpu_model(psenet_model, 2)
    ious = build_iou([0, 1], ['background', 'text'])

    summary_writer = tf.summary.create_file_writer("./logs/{}".format(localtime))

    # model save
    checkpoint = tf.train.Checkpoint(model=psenet_model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint, directory=MODEL_PATH.format(localtime),
                                         max_to_keep=max_to_keep)

    total_iou = tf.Variable(initial_value=0., dtype=tf.float32)

    # iter dataset
    data_train = Generator(TRAINING_DATA_PATH, batch_size=batch_size, istraining=True, num_classes=num_class,
                           mirror=False,reshape=shape)

    data_validate = Generator(VALIDATE_DATA_PATH, batch_size=batch_size, istraining=False, num_classes=num_class,
                              reshape=shape, mirror=False, scale=False, clip=False, trans_color=False)

    avg_loss = tf.keras.metrics.Mean(name="train_loss")
    val_avg_loss = tf.keras.metrics.Mean(name="val_loss")
    for epoch in range(1, N_EPOCH):
        with summary_writer.as_default():
            for X,Y in data_train.__next__():
                #print("debug:  " + "===="*30)
                #print( X.shape, Y.shape, "\n")      # (4, 640, 640, 3) (4, 160, 160, 6)
                loss = train_one_step(psenet_model, X, Y, optimizer)
                tf.summary.scalar("train_loss", loss, step=optimizer.iterations)
                tf.summary.scalar('learning_rate', LEARNING_RATE, step=optimizer.iterations)
                avg_loss.update_state(loss)
            print("[{} / {}] Mean train loss: {}".format(epoch, N_EPOCH, avg_loss.result()))
            avg_loss.reset_states()
            if (epoch - 1) % 10 == 0:
                saved_model_path = "./saved_models/{}".format(int(time.time()))
                tf.keras.experimental.export_saved_model(psenet_model, saved_model_path)
                saved_path = manager.save(checkpoint_number=epoch)
                print("Model saved to {}".format(saved_path))
                num_correct_samples = 0
                for X, Y in next(data_validate):
                    loss, accuracy = val_one_step(psenet_model, X, Y, total_iou)
                    val_avg_loss.update_state(loss)
                tf.summary.scalar("val_loss", val_avg_loss.result(), step=epoch)
                tf.summary.scalar("validate accuracy :", accuracy, step=epoch)
                print("[{} / {}] Mean val loss: {}".format(epoch, N_EPOCH, val_avg_loss.result()))
                print("[{} / {}] validate accuracy: {:.2f}".format(epoch, N_EPOCH, accuracy))
                val_avg_loss.reset_states()
