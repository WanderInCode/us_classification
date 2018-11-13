from keras.preprocessing.image import transform_matrix_offset_center, apply_transform, flip_axis
import numpy as np
import cv2


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


if __name__ == '__main__':
    # import cv2
    # import os
    # p = os.path.abspath('.') + '/small/train/'
    # img_p = p + '1_2.tif'
    # mask_p = p + '1_2_mask.tif'
    # img = cv2.imread(img_p, 0)
    # img = img[1:, 1:]
    # # img = cv2.merge((img, img, img))
    # mask = cv2.imread(mask_p, 0)
    # mask = mask[1:, 1:]
    # mask = cv2.merge((mask, mask, mask))

    # img_r, mask_r = random_rotation(img, mask, 20)
    # # img_r, mask_r = random_zoom(img, mask, (0.8, 1.2))
    # # img_r, mask_r = random_crop(img, mask, 463, 335)
    # cv2.imshow('img', img)
    # cv2.imshow('mask', mask)
    # cv2.imshow('img_r', img_r)
    # cv2.imshow('mask_r', mask_r)
    # cv2.waitKey()
    # import os
    # import cv2
    # p = os.path.abspath('.') + '/small/train/'
    # f = ['1_2.tif']
    # imgs_p = [p + i for i in f]
    # masks_p = [p + m.split('.')[0] + '_mask.tif' for m in f]
    # a = Augmentor((320, 224), 10, (0.8, 1.2), random_crop=(0.84, 0.84, 8))
    # batch_size = a.multiple
    # print(batch_size)
    # imgs, masks = a(imgs_p, masks_p, batch_size)
    # n = p + 'new/'
    # for idx, t in enumerate(zip(imgs, masks)):
    #     cv2.imwrite(str(idx + 1) + '.tif', t[0])
    #     cv2.imwrite(n + str(idx + 1) + '_mask.tif', t[1])
    import os
    import cv2
    import numpy as np
    p = os.path.abspath('.') + '/small/train/'
    f = ['1_1.tif', '1_2.tif', '1_3.tif']
    imgs_p = [p + i for i in f]
    imgs = [cv2.imread(p + i, 0) for i in f]
    # imgs = np.array(imgs)
    # print(imgs.shape)
    print(len(imgs))
    print(np.mean(imgs))
    print(np.std(imgs))
