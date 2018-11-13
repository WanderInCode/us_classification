import os
import numpy as np
from keras.preprocessing import sequence


def sample_split(folder, rate=0.2):
    file_list = os.listdir(folder)
    if len(file_list) == 0:
        raise 'this folder is empty!'
    imgs = [folder + f for f in file_list if 'mask' not in f]
    np.random.shuffle(imgs)
    masks = [m.split('.')[0] + '_mask.tif' for m in imgs]
    if len(imgs) != len(masks):
        raise 'the number of images and masks are not the same!'
    boundary = int(np.floor((len(imgs) * 0.2)))
    validate = (imgs[:boundary], masks[:boundary])
    train = (imgs[boundary:], masks[boundary:])
    return train, validate


class BatchGenerator(sequence.Sequence):

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


if __name__ == '__main__':
    path = os.path.abspath('.') + '/small/train/'
    a, b = sample_split(path)
    print(a)
    print('=========================================================')
    print(b)
