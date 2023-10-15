"""
Train the MobileNet V2 model
"""
import os, cv2
import sys
import argparse
import pandas as pd

from keras.optimizers import RMSprop
from keras.layers import Input
from keras.models import Model
from model import kl_divergence, correlation_coefficient,nss

from model_sia import testmodel
from config import *
from utilities import preprocess_images, postprocess_predictions

def main(argv):

    images_path = ''
    maps_path = ''
    gt_path = ''

    x_img_one = Input((shape_r, shape_c, 3))
    x_img_two = Input((shape_r, shape_c, 3))

    m = Model(inputs=[x_img_one, x_img_two], outputs=testmodel([x_img_one, x_img_two]))

    m.compile(RMSprop(lr=1e-6), loss=[kl_divergence, correlation_coefficient])

    output_folder = ''

    img_names = [f for f in os.listdir(gt_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    img_names.sort()

    nb_imgs_test = len(img_names)

    if nb_imgs_test % b_s != 0:
        print( "The number of test images should be a multiple of the batch size. Please change your batch size in config.py accordingly.")
        exit()

    print("Loading mobilenet weights")
    m.load_weights('')

    print("Predicting saliency maps for " + images_path)

    predictions = m.predict_generator(generator_test(b_s=b_s, imgs_one_path=images_path, imgs_two_path=maps_path), steps = nb_imgs_test/b_s)[0]

    for pred, name in zip(predictions, img_names):
        original_image = cv2.imread(gt_path + name, 0)
        res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
        cv2.imwrite(output_folder + '%s' % name, res.astype(int))

def generator_test(b_s, imgs_one_path, imgs_two_path):
    img_one = [imgs_one_path + f for f in os.listdir(imgs_one_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    img_two = [imgs_two_path + f for f in os.listdir(imgs_two_path) if f.endswith(('.png'))]

    img_one.sort()
    img_two.sort()

    counter = 0

    while True:
        X_img_one = preprocess_images(img_one[counter:counter + b_s], shape_r, shape_c)
        X_img_two = preprocess_images(img_two[counter:counter + b_s], shape_r, shape_c)

        yield [X_img_one, X_img_two]
        counter = (counter + b_s) % len(img_one)



if __name__ == '__main__':
    main(sys.argv)
