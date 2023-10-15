from __future__ import division
# from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Input, Reshape, AvgPool2D, Dropout, Conv2D, Softmax, BatchNormalization, Activation
from keras.layers import Lambda, merge, Conv2D, concatenate, UpSampling2D
from keras.layers.convolutional import Convolution2D
import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.layers import Input
from config import *


def repeat(x):
    return K.reshape(K.repeat(K.batch_flatten(x), nb_timestep), (b_s, nb_timestep, 512, shape_r_gt, shape_c_gt))


def repeat_shape(s):
    return (s[0], nb_timestep) + s[1:]


def upsampling(x):
    trans_img = tf.transpose(x, [0, 3, 1, 2])
    return trans_img


def upsampling_shape(s):
    trans_s = tf.transpose(s, [0, 3, 1, 2])
    return trans_s[:2] + (trans_s[2] * upsampling_factor, trans_s[3] * upsampling_factor, trans_s[3] * 1)

def reseting_shape(s):
    return s[:2] + (s[2] * reseting_factor, s[3] * reseting_factor)


# KL-Divergence Loss
def kl_divergence(y_true, y_pred):
    r_out = int(y_pred.shape[2])
    c_out = int(y_pred.shape[3])
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)),
                                                                   r_out, axis=-1)), c_out, axis=-1)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=2), axis=2)),
                                                                   r_out, axis=-1)), c_out, axis=-1)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=2), axis=2)),
                                                                   r_out, axis=-1)), c_out, axis=-1)
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    return  K.sum(K.sum(y_true * K.log((y_true / (y_pred + K.epsilon())) + K.epsilon()), axis=-1), axis=-1)


# Correlation Coefficient Loss
def correlation_coefficient(y_true, y_pred):
    r_out = int(y_pred.shape[2])
    c_out = int(y_pred.shape[3])
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)),
                                                                   r_out, axis=-1)), c_out, axis=-1)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=2), axis=2)),
                                                                   r_out, axis=-1)), c_out, axis=-1)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=2), axis=2)),
                                                                   r_out, axis=-1)), c_out, axis=-1)

    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    N = r_out * c_out
    sum_prod = K.sum(K.sum(y_true * y_pred, axis=2), axis=2)
    sum_x = K.sum(K.sum(y_true, axis=2), axis=2)
    sum_y = K.sum(K.sum(y_pred, axis=2), axis=2)
    sum_x_square = K.sum(K.sum(K.square(y_true), axis=2), axis=2)
    sum_y_square = K.sum(K.sum(K.square(y_pred), axis=2), axis=2)

    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))

    return -1 * num / den

# Normalized Scanpath Saliency Loss
def nss(y_true, y_pred):
    r_out = int(y_pred.shape[2])
    c_out = int(y_pred.shape[3])
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)),
                                                                   r_out, axis=-1)), c_out, axis=-1)
    y_pred /= max_y_pred
    y_pred_flatten = K.batch_flatten(y_pred)

    y_mean = K.mean(y_pred_flatten, axis=-1)
    y_mean = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_mean)),
                                                               r_out, axis=-1)), c_out, axis=-1)

    y_std = K.std(y_pred_flatten, axis=-1)
    y_std = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_std)),
                                                              r_out, axis=-1)), c_out, axis=-1)

    y_pred = (y_pred - y_mean) / (y_std + K.epsilon())

    return -(K.sum(K.sum(y_true * y_pred, axis=2), axis=2) / K.sum(K.sum(y_true, axis=2), axis=2))

def mse(y_true, y_pred):
    max_y = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)), shape_r_out, axis=-1)), shape_c_out, axis=-1)
    return K.mean(K.square((y_pred / max_y) - y_true) / (1 - y_true + 0.1))



