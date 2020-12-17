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
# refer from @git
# @Desc:
# 1).CustomSavingCallback： save weight and so on;
# 2).CustomLoaderCallback:  load weight;
# 3).LRTensorBoard: add lr to tensorboard;
# ==============================================================================
# LINT.IfChange

"""define model fit callbacks package"""


import os
import shutil
import pickle
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint


MODEL_WEIGHTS_FILENAME = 'weights.h5'
OPTIMIZER_WEIGHTS_FILENAME = 'optimizer_weights.pkl'
LEARNING_RATE_FILENAME = 'learning_rate.pkl'
LAYERS_FILENAME = 'architecture.json'
EPOCH_FILENAME = 'epoch.pkl'
FOLDER_SAVED_MODEL = 'saving'


class CustomSavingCallback(Callback):
    """
    Callback to save weights, architecture, and optimizer at the end of training.
    Inspired by `ModelCheckpoint`.

    :ivar output_dir: path to the folder where files will be saved
    :vartype output_dir: str
    :ivar saving_freq: save every `n` epochs
    :vartype saving_freq: int
    :ivar save_best_only: wether to save a model if it is best thant the last saving
    :vartype save_best_only: bool
    :ivar keep_max_models: number of models to keep, the older ones will be deleted
    :vartype keep_max_models: int
    """
    def __init__(self,
                 output_dir: str,
                 saving_freq:int,
                 save_best_only: bool=False,
                 keep_max_models:int=5):
        super(CustomSavingCallback, self).__init__()

        self.saving_dir = output_dir
        self.saving_freq = saving_freq
        self.save_best_only = save_best_only
        self.keep_max_models = keep_max_models

        self.epochs_since_last_save = 0

        self.monitor = 'val_loss'
        self.monitor_op = np.less
        self.best_value = np.Inf # todo: when restoring model we could also restore val_loss and metric

    def on_epoch_begin(self,
                       epoch,
                       logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self,
                     epoch,
                     logs=None):

        self.logs = logs
        self.epochs_since_last_save += 1

        if self.epochs_since_last_save == self.saving_freq:
            self._export_model(logs)
            self.epochs_since_last_save = 0

    def on_train_end(self,
                     logs=None):
        self._export_model(self.logs)
        self.epochs_since_last_save = 0


    def _export_model(self, logs):
        timestamp = str(int(time.time()))
        folder = os.path.join(self.saving_dir, timestamp)

        if self.save_best_only:
            current_value = logs.get(self.monitor)
            if current_value is None:
                current_value = 0

            if self.monitor_op(current_value, self.best_value):
                print('\n{} improved from {:0.5f} to {:0.5f},'
                      ' saving model to {}'.format(self.monitor, self.best_value,
                                                   current_value, folder))
                self.best_value = current_value

            else:
                print('\n{} did not improve from {:0.5f}'.format(self.monitor, self.best_value))
                return

        os.makedirs(folder)

        # save architecture
        model_json = self.model.to_json()
        with open(os.path.join(folder, LAYERS_FILENAME), 'w') as f:
            json.dump(model_json, f)

        # model weights
        self.model.save_weights(os.path.join(folder, MODEL_WEIGHTS_FILENAME))

        # optimizer weights
        optimizer_weights = tf.keras.backend.batch_get_value(self.model.optimizer.weights)
        with open(os.path.join(folder, OPTIMIZER_WEIGHTS_FILENAME), 'wb') as f:
            pickle.dump(optimizer_weights, f)

        # learning rate
        learning_rate = self.model.optimizer.learning_rate
        with open(os.path.join(folder, LEARNING_RATE_FILENAME), 'wb') as f:
            pickle.dump(learning_rate, f)

        # n epochs
        epoch = self._current_epoch + 1
        with open(os.path.join(folder, EPOCH_FILENAME), 'wb') as f:
            pickle.dump(epoch, f)

        self._clean_exports()

    def _clean_exports(self):
        timestamp_folders = [int(f) for f in os.listdir(self.saving_dir)]
        timestamp_folders.sort(reverse=True)

        if len(timestamp_folders) > self.keep_max_models:
            folders_to_remove = timestamp_folders[self.keep_max_models:]
            for f in folders_to_remove:
                shutil.rmtree(os.path.join(self.saving_dir, str(f)))



class CustomLoaderCallback(Callback):
    """
    Callback to load necessary weight and parameters for training, evaluation and prediction.

    :ivar loading_dir: path to directory to save logs
    :vartype loading_dir: str
    """
    def __init__(self,
                 loading_dir: str):
        super(CustomLoaderCallback, self).__init__()

        self.loading_dir = loading_dir

    def set_model(self, model):
        self.model = model

        print('-- Loading ', self.loading_dir)
        # Load model weights
        self.model.load_weights(os.path.join(self.loading_dir, MODEL_WEIGHTS_FILENAME))

        # Load optimizer params
        with open(os.path.join(self.loading_dir, OPTIMIZER_WEIGHTS_FILENAME), 'rb') as f:
            optimizer_weights = pickle.load(f)
        with open(os.path.join(self.loading_dir, LEARNING_RATE_FILENAME), 'rb') as f:
            learning_rate = pickle.load(f)

        # Set optimizer params
        self.model.optimizer.learning_rate.assign(learning_rate)
        self.model._make_train_function()
        self.model.optimizer.set_weights(optimizer_weights)


class LRTensorBoard(TensorBoard):
    """
    Adds learning rate to TensorBoard scalars.
    # From https://github.com/keras-team/keras/pull/9168#issuecomment-359901128
    :ivar logdir: path to directory to save logs
    :vartype logdir: str
    """

    def __init__(self,
                 log_dir: str,
                 **kwargs):  # add other arguments to __init__ if you need
        super(LRTensorBoard, self).__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self,
                     epoch,
                     logs=None):
        logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        super(LRTensorBoard, self).on_epoch_end(epoch, logs)

def LRCallback(factor:int,
               patience:int,
               cooldown:int,
               min_lr=1e-8,
               verbose=1,
               *args, **kwargs):
    '''
    When the index stops improving, the learning rate will be reduced.
    Once learning stops, the model usually reduces the learning rate by 2-10 times.
    This callback monitors the number.
    If you don't see an improvement in the number of 'patients' in epoch, the learning rate will decrease.

    '''
    return tf.keras.callbacks.ReduceLROnPlateau(factor=factor,
                                                 patience=patience,
                                                 cooldown=cooldown,
                                                 min_lr=min_lr,
                                                 verbose=verbose)

def ESCallback(min_delta: float,
               patience: int,
               verbose =1,
               *args, **kwargs):
    '''
    Stop training when monitoring quantity stops improving.

    :return:
    '''
    return tf.keras.callbacks.EarlyStopping(min_delta=min_delta,
                                             patience=patience,
                                             verbose=verbose)

def TensorBoardCallback(log_dir: str, profile_batch=0):
    '''
    add le to Tensorboard
    :param logdir:
    :return:
    '''
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir,profile_batch=0)

def CheckpointSaveCallback(MODEL_PATH):
    '''
    save model callbacks
    Parameters
    ----------
    MODEL_PATH : model save path

    Returns object save model
    -------

    '''
    checkpoint = ModelCheckpoint(filepath=MODEL_PATH,save_weights_only=True)
    return checkpoint