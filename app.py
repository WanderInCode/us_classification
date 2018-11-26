from keras_preprocessing.image import transform_matrix_offset_center, flip_axis
from keras.layers import Input, Convolution2D, MaxPooling2D, \
    BatchNormalization, Activation, GlobalAveragePooling2D, Dense
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model
from keras import backend as K
from keras import layers
from keras.utils import Sequence
import numpy as np
import cv2
import scipy.ndimage as ndi
import os


def sample_split(files_list, rate=0.2):
    imgs = files_list[0]
    masks = files_list[1]
    state = np.random.get_state()
    np.random.shuffle(imgs)
    np.random.set_state(state)
    np.random.shuffle(masks)
    boundary = int(np.floor((len(imgs) * 0.2)))
    validate = (imgs[:boundary], masks[:boundary])
    train = (imgs[boundary:], masks[boundary:])
    return train, validate


class BatchGenerator(Sequence):

    def __init__(self, imgs, masks, batch_size, augmentor, name=None):
        self.imgs = imgs
        self.masks = masks
        self.batch_size = batch_size
        self.augmentor = augmentor
        if name:
            self.name = name
        else:
            self.name = ''

    def __len__(self):
        return int(np.ceil(len(self.imgs) / float(self.batch_size)))

    def __getitem__(self, idx):
        multiple = int(np.ceil(self.batch_size /
                               float(self.augmentor.multiple)))
        batch_imgs = self.imgs[idx * multiple:(idx + 1) * multiple]
        batch_masks = self.masks[idx * multiple:(idx + 1) * multiple]
        return self.augmentor(batch_imgs, batch_masks, self.batch_size)

    def on_epoch_end(self):
        l = np.vstack((self.imgs, self.masks))
        l = l.T
        np.random.shuffle(l)
        self.imgs = l[:, 0].tolist()
        self.masks = l[:, 1].tolist()


def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=1,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def random_zoom(x, y, zoom_range, row_index=0, col_index=1, channel_index=2,
                fill_mode='nearest', cval=0.):
    if len(zoom_range) != 2:
        raise Exception('zoom_range should be a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_index, fill_mode, cval)
    return x, y


def random_rotation(x, y, rg, row_index=0, col_index=1, channel_index=2,
                    fill_mode='nearest', cval=0.):
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_index, fill_mode, cval)
    return x, y


def random_crop(x, y, w_rate, h_rate, row_index=0, col_index=1):
    h, w = x.shape[row_index], x.shape[col_index]
    w_crop = int(np.ceil(w * w_rate))
    h_crop = int(np.ceil(h * h_rate))
    delta_w = w - w_crop
    delta_h = h - h_crop
    w_start = np.random.randint(0, delta_w)
    h_start = np.random.randint(0, delta_h)
    x_crop = x[h_start: h_start + h_crop, w_start: w_start + w_crop, :]
    y_crop = y[h_start: h_start + h_crop, w_start: w_start + w_crop, :]
    return x_crop, y_crop


class Augmentor(object):

    def __init__(self, new_size, rotate, zoom, h_flip=True, v_flip=False, crop=None, mul_channel=None, val=False):
        if len(new_size) != 2 or (not isinstance(new_size, list) and not isinstance(new_size, tuple)):
            raise Exception('new_size must be list or tuple, and have 2 items')
        if len(zoom) != 2 or (not isinstance(zoom, list) and not isinstance(zoom, tuple)):
            raise Exception(
                'random_zoom must be list or tuple, and have 2 items')

        self.val = val
        self.new_size = new_size
        self.rotate = rotate
        self.zoom = zoom
        self.multiple = 3
        self.h_flip = h_flip
        self.v_flip = v_flip
        if h_flip:
            self.multiple += 1
        if v_flip:
            self.multiple += 1
        # (w_rate, h_rate, times) (0.84, 0.84, 8)
        self.crop = crop
        if crop:
            self.multiple = self.multiple * crop[2] + 1
        if val:
            self.multiple = 1
        self.mul_channel = mul_channel

    def __call__(self, imgs_p, masks_p, batch_size):
        return img_augment(imgs_p, masks_p, self.new_size, self.rotate, self.zoom, self.h_flip, self.v_flip, self.crop, batch_size, self.val)


def class_mask(imgs, masks, new_size, crop=None):
    img_batches = []
    mask_batches = []
    if crop:
        width_rate = crop[0]
        height_rate = crop[1]
        times = crop[2]

    for i, m in zip(imgs, masks):
        img_batches.append(cv2.resize(
            i, new_size, interpolation=cv2.INTER_AREA)[..., np.newaxis])
        if m.sum() > 100:
            mask_batches.append([1, 0])
        else:
            mask_batches.append([0, 1])

        if crop:
            for _ in range(times):
                img_c, mask_c = random_crop(i, m, width_rate, height_rate)
                # crop_imgs.append(cv2.resize(
                #     img_c, new_size, interpolation=cv2.INTER_AREA))
                img_batches.append(cv2.resize(
                    img_c, new_size, interpolation=cv2.INTER_AREA)[..., np.newaxis])
                if mask_c.sum() > 100:
                    mask_batches.append([1, 0])
                else:
                    mask_batches.append([0, 1])
    return img_batches, mask_batches


def img_augment(imgs_p, masks_p, new_size, rotate, zoom, h_flip, v_flip, crop,
                batch_size, val):
    # print(imgs_p)
    imgs = [cv2.imread(p, 0)[1:, 1:] for p in imgs_p]
    masks = [cv2.imread(p, 0)[1:, 1:] for p in masks_p]
    imgs_container = []
    masks_container = []
    for img, mask in zip(imgs, masks):
        # img_o = cv2.merge((img, img, img))
        img_o = img[..., np.newaxis]
        mask = mask / 255.0
        mask_o = mask[..., np.newaxis]
        imgs_container.append(img_o)
        masks_container.append(mask_o)

        img_r, mask_r = random_rotation(img_o, mask_o, rotate)
        imgs_container.append(img_r)
        masks_container.append(mask_r)

        img_z, mask_z = random_zoom(img_o, mask_o, zoom)
        imgs_container.append(img_z)
        masks_container.append(mask_z)

        if h_flip:
            img_f = flip_axis(img_o, 1)
            mask_f = flip_axis(mask_o, 1)
            imgs_container.append(img_f)
            masks_container.append(mask_f)

        if v_flip:
            img_f = flip_axis(img_o, 0)
            mask_f = flip_axis(mask_o, 0)
            imgs_container.append(img_f)
            masks_container.append(mask_f)

    img_batches, mask_batches = class_mask(
        imgs_container,
        masks_container,
        new_size,
        crop)

    if batch_size == 0:
        batch_size = len(img_batches)
    return np.array(img_batches[0:batch_size]), np.array(mask_batches[0:batch_size])


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
    x = Dense(classes, activation='softmax')(x)
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


def train_classifier(classes, path, epoch, lr):
    model = res34(classes)
    model.compile(optimizer=Adam(lr=lr),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    train, validate = sample_split(path)
    a = Augmentor((320, 224), 10, (0.8, 1.2), crop=None)
    t = BatchGenerator(train[0], train[1], 32, a)
    vals = img_augment(validate[0], validate[1], (320, 224),
                       10, (0.8, 1.2), True, False, None, 0, False)

    p = os.path.abspath('.')

    model_checkpoint = ModelCheckpoint(
        os.path.join(p, 'lr_00001_weights_{epoch:02d}_{val_acc:.2f}.hdf5'),
        monitor='val_acc', save_best_only=True)
    callbacks = [model_checkpoint]

    model.fit_generator(t, epochs=epoch, verbose=1, callbacks=callbacks,
                        validation_data=vals)


def main():
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
    train_classifier(2, [train_pic_names, train_mask_names], 100, 0.0001)


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


main()
