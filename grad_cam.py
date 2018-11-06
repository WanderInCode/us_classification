from keras.models import Sequential
import keras.backend as K
import numpy as np


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
