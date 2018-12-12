from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Input, Convolution2D, MaxPooling2D, \
    BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras import backend as K
from keras import layers
from keras.utils import Sequence
import numpy as np
import cv2
import scipy.ndimage as ndi
import os
import json


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(filters1, kernel_size, padding='same',
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(filters2, kernel_size, padding='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2)):
    filters1, filters2 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(filters1, kernel_size, padding='same', strides=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(filters2, kernel_size, padding='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    shortcut = Convolution2D(filters2, (1, 1), strides=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def res34(classes, input_shape=(224, 320, 1)):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    img = Input(shape=input_shape)
    x = Convolution2D(64, (7, 7), strides=(2, 2),
                      padding='same', name='conv1')(img)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    # print(encoder1.shape)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    # print(x.shape)
    x = conv_block(x, 3, [64, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64], stage=2, block='b')
    x = identity_block(x, 3, [64, 64], stage=2, block='c')

    x = conv_block(x, 3, [128, 128], stage=3, block='a')
    x = identity_block(x, 3, [128, 128], stage=3, block='b')
    x = identity_block(x, 3, [128, 128], stage=3, block='c')
    x = identity_block(x, 3, [128, 128], stage=3, block='d')

    x = conv_block(x, 3, [256, 256], stage=4, block='a')
    x = identity_block(x, 3, [256, 256], stage=4, block='b')
    x = identity_block(x, 3, [256, 256], stage=4, block='c')
    x = identity_block(x, 3, [256, 256], stage=4, block='d')
    x = identity_block(x, 3, [256, 256], stage=4, block='e')
    x = identity_block(x, 3, [256, 256], stage=4, block='f')

    x = conv_block(x, 3, [512, 512], stage=5, block='a')
    x = identity_block(x, 3, [512, 512], stage=5, block='b')
    x = identity_block(x, 3, [512, 512], stage=5, block='c')

    x = GlobalAveragePooling2D()(x)
    # do not need Flatten layer!
    # x = Flatten()(x)
    # cam weight
    # x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax')(x)
    m = Model(img, x)

    return m


def res18(classes, input_shape=(224, 320, 1)):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    img = Input(shape=input_shape)
    x = Convolution2D(64, (7, 7), strides=(2, 2),
                      padding='same', name='conv1')(img)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    # print(encoder1.shape)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    # print(x.shape)
    x = conv_block(x, 3, [64, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64], stage=2, block='b')

    x = conv_block(x, 3, [128, 128], stage=3, block='a')
    x = identity_block(x, 3, [128, 128], stage=3, block='b')

    x = conv_block(x, 3, [256, 256], stage=4, block='a')
    x = identity_block(x, 3, [256, 256], stage=4, block='b')

    x = conv_block(x, 3, [512, 512], stage=5, block='a')
    x = identity_block(x, 3, [512, 512], stage=5, block='b')

    x = GlobalAveragePooling2D()(x)
    # do not need Flatten layer!
    # x = Flatten()(x)
    # cam weight
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='sigmoid')(x)
    m = Model(img, x)

    return m


def grad_cam(model, images, layer_name):
    loss = K.max(model.output, axis=-1, keepdims=True)
    conv_layer = model.get_layer(layer_name).output
    grad = K.gradients(loss, conv_layer)[0]
    mean_grad = K.mean(grad, axis=(1, 2))
    f = K.function([model.input], [conv_layer, mean_grad])
    conv_out_val, grad_val = f(images)
    weight_maps = np.zeros(conv_out_val.shape[:3])
    for b in range(conv_out_val.shape[0]):
        for c in range(conv_out_val.shape[3]):
            weight_maps[b] += conv_out_val[b, :, :, c] * grad_val[b, c]
    return weight_maps


train_files_path = '/kaggle/input/filename/train_pic.txt'
train_pic_prefix = '/kaggle/input/ultrasound-nerve-segmentation/train/'


def img_reader(new_size=(320, 224)):
    trains = []
    labels = []
    with open(train_files_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            pic_path = train_pic_prefix + line.strip()
            mask_path = train_pic_prefix + \
                line.strip().split('.')[0] + '_mask.tif'
            pic = cv2.imread(pic_path, 0)[1:, 1:]
            pic = cv2.resize(pic, new_size, interpolation=cv2.INTER_AREA)
            pic = pic[..., np.newaxis]
            trains.append(pic)
            mask = cv2.imread(mask_path, 0)
            mask = mask / 255.0
            if np.sum(mask) > 100:
                # labels.append([1, 0])
                labels.append([1])
            else:
                # labels.append([0, 1])
                labels.append([0])
    trains = np.array(trains)
    return trains, labels


def train_classifier(classes, path, epoch, lr, weights_path=None):
    model = res18(classes)
    if weights_path:
        model.load_weights(weights_path)
    model.compile(optimizer=Adam(lr=float(lr)),
                  loss='binary_crossentropy', metrics=['accuracy'])
    trains, labels = img_reader()
    generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10,
        shear_range=0.2,
        zoom_range=(0.8, 1.2),
        rescale=1./255,
        horizontal_flip=True,
        validation_split=0.2)
    generator.fit(trains)
    train_flow = generator.flow(
        trains, labels, shuffle=True, subset='training')
    validate_flow = generator.flow(
        trains, labels, shuffle=True, subset='validation')

    STEP_SIZE_TRAIN = train_flow.n//train_flow.batch_size
    STEP_SIZE_VALID = validate_flow.n//validate_flow.batch_size

    p = os.path.abspath('.')
    weights_name = 'lr_0' + lr[2:] + '_weights_{epoch:02d}_{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(
        os.path.join(p, weights_name),
        monitor='val_acc', save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10)
    callbacks = [model_checkpoint, reduce_lr]

    history = model.fit_generator(train_flow,
                                  epochs=epoch,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  verbose=1,
                                  callbacks=callbacks,
                                  validation_data=validate_flow,
                                  validation_steps=STEP_SIZE_VALID)
    with open('history.json', 'w') as f:
        json.dump(history.history, f)


def main(classes, epoch, lr, weights_path=None):
    train_pic_names = []
    train_mask_names = []
    with open(train_files_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            train_pic_names.append(train_pic_prefix + line.strip())
            mask_pic_name = line.strip().split('.')[0] + '_mask.tif'
            train_mask_names.append(train_pic_prefix + mask_pic_name)
    train_classifier(
        classes, [train_pic_names, train_mask_names], epoch, lr, weights_path)


def test():
    train_files_path = '/kaggle/input/filename/train_pic.txt'
    train_pic_names = []
    train_mask_names = []
    with open(train_files_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            train_pic_prefix = '/kaggle/input/ultrasound-nerve-segmentation/train/'
            train_pic_names.append(train_pic_prefix + line.strip())
            mask_pic_name = line.strip().split('.')[0] + '_mask.tif'
            train_mask_names.append(train_pic_prefix + mask_pic_name)
    train, validate = sample_split([train_pic_names, train_mask_names])
    print(validate)


main(2, 100, '0.0001')
