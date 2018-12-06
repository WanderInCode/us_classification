from .model import res34
import cv2
import numpy as np
import csv


def predict(model, weight_path, img_name, path_prefix, img_size):
    model.load_weights(weight_path)
    imgs = []
    masks = []
    for p in img_name:
        # print(path_prefix + p)
        img = cv2.imread(path_prefix + p, 0)[1:, 1:]
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        img = img[..., np.newaxis]
        imgs.append(img)
    imgs = np.array(imgs)
    predicts = model.predict(imgs, verbose=1)
    f = open('statistic.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['file_name', 'class', 'predict',
                     'pro', 'TP', 'FN', 'FP', 'TN'])
    r = 0
    for idx, classes in enumerate(predicts):
        clss = np.argmax(classes, axis=-1)
        if clss == 0:
            p = 1
        else:
            p = 0
        mask_name = img_name[idx].split('.')[0] + '_mask.tif'
        mask = cv2.imread(path_prefix + mask_name, 0)
        mask = mask / 255.0
        if mask.sum() > 100:
            c = 1
        else:
            c = 0
        if p == c:
            r += 1
        writer.writerow([img_name[idx], str(c), str(p),
                         str(classes), 0, 0, 0, 0])
    acc = r / len(predicts)
    writer.writerow(['', '', '', '', '', '', '', acc])
    f.close()
