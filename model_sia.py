from __future__ import division
# from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Input, Reshape, AvgPool2D, Dropout, Conv2D, Softmax, BatchNormalization, Activation, MaxPooling2D
from keras.layers import Lambda, merge, Conv2D, concatenate, UpSampling2D
from keras.layers.convolutional import Convolution2D, AtrousConvolution2D
from keras.models import Model
import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.layers import Input
from config import *
# from shufflenetv2 import ShuffleNetV2
from efficientnet import EfficientNetB5



def repeat(x):
    return K.reshape(K.repeat(K.batch_flatten(x), nb_timestep), (b_s, nb_timestep, 512, shape_r_gt, shape_c_gt))


def repeat_shape(s):
    return (s[0], nb_timestep) + s[1:]


def upsampling(x):
    trans_img = tf.transpose(x, [0, 3, 1, 2])
    return trans_img

def upsampling1(x):
    trans_img = tf.transpose(x, [0, 1, 2, 3])
    return trans_img

def upsampling_shape(s):
    trans_s = tf.transpose(s, [0, 3, 1, 2])
    return trans_s[:2] + (trans_s[2] * upsampling_factor, trans_s[3] * upsampling_factor, trans_s[3] * 1)

def reseting_shape(s):
    return s[:2] + (s[2] * reseting_factor, s[3] * reseting_factor)



def testmodel(sharein):
    share_efficient1 = EfficientNetB5(input_tensor=sharein[0], input_shape=(shape_r, shape_c, 3), sianum='sia0')
    share_efficient2 = EfficientNetB5(input_tensor = sharein[1],input_shape = (shape_r, shape_c, 3), sianum='sia1')

    drop00 = share_efficient1.get_layer("sia0_block7c_add").output
    drop10 = share_efficient2.get_layer("sia1_block7c_add").output

    merge00 = concatenate([drop00, drop10], axis=3)

    up1 = Conv2D(224, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(merge00))

    drop01 = share_efficient1.get_layer("sia0_block5j_add").output
    drop11 = share_efficient2.get_layer("sia1_block5j_add").output

    merge01 = concatenate([up1, drop01, drop11], axis=3)


    atrous01 = AtrousConvolution2D(224, 5, 5, border_mode='same', activation='relu', atrous_rate=(2, 2))(merge01)

    up2 = Conv2D(80, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(atrous01))

    drop02 = share_efficient1.get_layer("sia0_block3g_add").output
    drop12 = share_efficient2.get_layer("sia1_block3g_add").output

    merge02 = concatenate([up2, drop02, drop12], axis=3)

    atrous02 = AtrousConvolution2D(80, 5, 5, border_mode='same', activation='relu', atrous_rate=(2, 2))(merge02)

    up3 = Conv2D(48, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(atrous02))

    drop03 = share_efficient1.get_layer("sia0_block2g_add").output
    drop13 = share_efficient2.get_layer("sia1_block2g_add").output

    merge03 = concatenate([up3, drop03, drop13], axis=3)

    atrous03 = AtrousConvolution2D(48, 5, 5, border_mode='same', activation='relu', atrous_rate=(2, 2))(merge03)

    up4 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(atrous03))

    drop04 = share_efficient1.get_layer("sia0_block1d_add").output
    drop14 = share_efficient2.get_layer("sia1_block1d_add").output

    merge04 = concatenate([up4, drop04, drop14], axis=3)
    atrous04 = AtrousConvolution2D(32, 5, 5, border_mode='same', activation='relu', atrous_rate=(2, 2))(merge04)

    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(atrous04)#128
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)


    efficient_outs = Conv2D(1, 1, padding='same', activation='relu')(conv9)

    outputs_up = Lambda(upsampling, (1, 240, 320))(efficient_outs)

    return [outputs_up, outputs_up]

