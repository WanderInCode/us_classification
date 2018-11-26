from keras import backend as K
from augment import Augmentor, img_augment
from keras.models import Model, model_from_json
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
import numpy as np
import os
import platform
from keras.losses import categorical_crossentropy

import tensorflow as tf


def train(epoch, lr):
    # u_net = get_u_net(freeze=False, set_trainable=set_network)

    u_net = res_unet()

    # u_net = get_unet()

    u_net.compile(optimizer=Adam(lr=lr),
                  loss=dice_coef_loss, metrics=[dice_coef])

    if not os.path.exists('model_architecture.json'):
        json = u_net.to_json()
        with open('model_architecture.json', 'w') as json_file:
            json_file.write(json)

    train, validate = sample_split(path)
    a = Augmentor((320, 224), 10, (0.8, 1.2), crop=None)
    # b = Augmentor((320, 224), 10, (0.8, 1.2), val=True)
    vals = img_augment(validate[0], validate[1], (320, 224),
                       10, (0.8, 1.2), True, False, None, 0, False)
    t = BatchGenerator(train[0], train[1], 32, a)
    # vals = BatchGenerator(validate[0], validate[1], 32, b)
    p = os.path.abspath('.')

    # board = TensorBoard(log_dir=p+'\\f8', histogram_freq=0,
    #                     write_graph=True, write_images=True)
    model_checkpoint = ModelCheckpoint(
        os.path.join(p, 'lr_00001_weights_{epoch:02d}.hdf5'),
        monitor='val_loss', save_best_only=False)
    callbacks = [model_checkpoint]

    u_net.fit_generator(t, epochs=epoch, verbose=1, callbacks=callbacks,
                        validation_data=vals)


def train_again(epoch, lr):
    json_file = open('model_architecture.json', 'r')
    json = json_file.read()
    u_net = model_from_json(json)
    u_net.load_weights('lr_0000002_weights_04.hdf5')
    u_net.compile(optimizer=Adam(lr=lr),
                  loss=dice_coef_loss, metrics=[dice_coef])
    train, validate = sample_split(path)
    a = Augmentor((320, 224), 10, (0.8, 1.2), crop=None)
    # b = Augmentor((320, 224), 10, (0.8, 1.2), val=True)
    vals = img_augment(validate[0], validate[1], (320, 224),
                       10, (0.8, 1.2), True, False, None, 0, False)
    t = BatchGenerator(train[0], train[1], 32, a)
    # vals = BatchGenerator(validate[0], validate[1], 32, b)
    p = os.path.abspath('.')

    # board = TensorBoard(log_dir=p+'\\f8', histogram_freq=0,
    #                     write_graph=True, write_images=True)
    model_checkpoint = ModelCheckpoint(
        os.path.join(p, 'lr_00000002_weights_{epoch:02d}.hdf5'),
        monitor='val_loss', save_best_only=False)
    # callbacks = [model_checkpoint, board]
    callbacks = [model_checkpoint]

    u_net.fit_generator(t, epochs=epoch, verbose=1, callbacks=callbacks,
                        validation_data=vals)


def train_classifier(epoch, lr, model, weight):
    json_file = open(model, 'r')
    json = json_file.read()
    res_net = model_from_json(json)
    res_net.load_weights(weight, by_name=True)
    for layer in res_net.layers:
        layer.trainable = False
    last_layer = res_net.layers['act5c_branch2b']

    res_net.compile(optimizer=Adam(lr=lr),
                    loss='categorical_crossentropy')
    train, validate = sample_split(path)
    a = Augmentor((320, 224), 10, (0.8, 1.2), crop=None)
    vals = img_augment(validate[0], validate[1], (320, 224),
                       10, (0.8, 1.2), True, False, None, 0, False)
    t = BatchGenerator(train[0], train[1], 32, a)
    p = os.path.abspath('.')

    model_checkpoint = ModelCheckpoint(
        os.path.join(p, 'lr_00000002_weights_{epoch:02d}.hdf5'),
        monitor='val_loss', save_best_only=False)
    callbacks = [model_checkpoint]

    res_net.fit_generator(t, epochs=epoch, verbose=1, callbacks=callbacks,
                          validation_data=vals)


if __name__ == '__main__':
    # train(50, 1e-4)
    train_again(30, 2e-7)
