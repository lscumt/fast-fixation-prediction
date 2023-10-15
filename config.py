# coding=UTF-8
# -*- coding:UTF-8 -*-
#########################################################################
# MODEL PARAMETERS														#
#########################################################################

# batch size
b_s = 1

steps_perepoch = 10000
# number of rows of input images224 456
shape_r = 480
# number of cols of input images
shape_c = 640
# number of rows of input images224 456
actshape_r = 30
# number of cols of input images
actshape_c = 40
# number of rows of model outputs
shape_r_out = 240
# number of cols of model outputs
shape_c_out = 320
# final upsampling factor
upsampling_factor = 4
# reset factor
reseting_factor = 4
# number of epochs
nb_epoch = 25
# number of timestep
nb_timestep = 4
# number of learned priors
nb_gaussian = 16

#########################################################################
# TRAINING SETTINGS										            	#
#########################################################################
# path of training images
img_train_one_path = ''
# path of training maps
img_train_two_path = ''
# path of training groundtruth
img_train_three_path = ''
# path of training groundtruth
gt_train_path = ''

# number of training images
nb_imgs_train = 10000
# path of validation images
img_val_one_path = ''
# path of validation maps
img_val_two_path = ''
# path of validation fixation
img_val_three_path = ''
# path of validation groundtruth
gt_val_path = ''
# number of validation images
nb_imgs_val = 5000
