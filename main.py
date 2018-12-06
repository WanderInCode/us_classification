from ResCls.statistic import predict
from ResCls.model import res34
import os


def main():
    model = res34(2)

    path_prefix = os.path.abspath('.') + '/train/'
    weight_path = os.path.abspath('.') + '/lr_0000001_weights_02_0.85.hdf5'
    f = open('test_pic.txt', 'r')
    img_names = []
    for line in f:
        if line.strip() == '':
            continue
        img_names.append(line.strip())
    img_size = (320, 224)
    predict(model, weight_path, img_names, path_prefix, img_size)


if __name__ == '__main__':
    main()
