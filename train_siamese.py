"""
Train the MobileNet V2 model
"""
import os
import sys
import argparse

from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Input
from keras.models import Model
from model import kl_divergence, correlation_coefficient,nss
from model_sia import testmodel
from config import *
from utilities import preprocess_images, preprocess_maps
from Loss_accuracy import LossHistory

import keras.backend as K


def main(argv):
    parser = argparse.ArgumentParser()

    x_img_one = Input((shape_r, shape_c, 3))
    x_img_two = Input((shape_r, shape_c, 3))

    m = Model(inputs=[x_img_one, x_img_two], outputs=testmodel([x_img_one, x_img_two]))

    # for layer in m.layers:
    #     layer.trainable = False
    # # 或者使用如下方法冻结所有层
    # # model.trainable = False
    # for i in range(1, 22):
    #     m.layers[-i].trainable = True

    m.compile(RMSprop(lr=1e-4), loss=[kl_divergence,correlation_coefficient])

    logs_loss = LossHistory()

    def scheduler(epoch):
        lr = K.get_value(m.optimizer.lr)
        if lr % 2 != 0:
            K.set_value(m.optimizer.lr, lr * 0.1)
        return K.get_value(m.optimizer.lr)

    lr_new = LearningRateScheduler(scheduler)

    if nb_imgs_train % b_s != 0 or nb_imgs_val % b_s != 0:
        print("The number of training and validation images should be a multiple of the batch size. Please change your batch size in config.py accordingly.")
        exit()

    hist = m.fit_generator(generator(b_s=b_s), steps_per_epoch=nb_imgs_train/b_s,verbose=1, epochs=nb_epoch,
                    validation_data=generator(b_s=b_s, phase_gen='val'), validation_steps=nb_imgs_val/b_s,
                    callbacks=[EarlyStopping(patience=10),lr_new,
                               ModelCheckpoint('savePath/weights.{epoch:02d}-{val_loss:.4f}.hdf5', save_best_only=True), logs_loss])


def generator(b_s, phase_gen='train'):
    if phase_gen == 'train':
        img_one = [img_train_one_path + f for f in os.listdir(img_train_one_path) if f.endswith('.jpg')]
        img_two = [img_train_two_path + f for f in os.listdir(img_train_two_path) if f.endswith('.png')]
        fixs = [img_train_three_path + f for f in os.listdir(img_train_three_path) if f.endswith('.jpg')]
        maps = [gt_train_path + f for f in os.listdir(gt_train_path) if f.endswith('.jpg')]

    elif phase_gen == 'val':
        img_one = [img_val_one_path + f for f in os.listdir(img_val_one_path) if f.endswith('.jpg')]
        img_two = [img_val_two_path + f for f in os.listdir(img_val_two_path) if f.endswith('.png')]
        fixs = [img_val_three_path + f for f in os.listdir(img_val_three_path) if f.endswith('.jpg')]
        maps = [gt_val_path + f for f in os.listdir(gt_val_path) if f.endswith('.jpg')]
    else:
        raise NotImplementedError

    img_one.sort()
    img_two.sort()
    fixs.sort()
    maps.sort()

    counter = 0

    while True:

        Y = preprocess_maps(maps[counter:counter + b_s], shape_r_out, shape_c_out)
        Y_fix = preprocess_maps(fixs[counter:counter + b_s], shape_r_out, shape_c_out)
        X_img_one = preprocess_images(img_one[counter:counter + b_s], shape_r, shape_c)
        X_img_two = preprocess_images(img_two[counter:counter + b_s], shape_r, shape_c)


        yield [X_img_one, X_img_two], [Y,Y]
        counter = (counter + b_s) % len(img_one)


if __name__ == '__main__':
    main(sys.argv)
