import numpy as np
from keras.layers import Layer
from keras import backend as K
import tensorflow as tf
import keras
import math

class MaxBlurPooling1D(Layer):

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.avg_kernel = None
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(MaxBlurPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 2:
            bk = np.array([1, 1])
        elif self.kernel_size == 3:
            bk = np.array([1, 2, 1])
        elif self.kernel_size == 5:
            bk = np.array([1, 4, 6, 4, 1])
        else:
            if (math.log(self.kernel_size - 1, 2) / math.ceil(math.log(self.kernel_size - 1, 2))) != 1:
                raise ValueError
            else:
                blur_kernel = np.array([1, 1])
                for i in range(int(math.log(self.kernel_size - 1, 2))):
                    blur_kernel = np.convolve(blur_kernel, blur_kernel)

        bk = bk / np.sum(bk)
        bk = np.repeat(bk, input_shape[2])
        bk = np.reshape(bk, (self.kernel_size, 1, input_shape[2], 1))
        blur_init = keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, 1, input_shape[2], 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(MaxBlurPooling1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        x = tf.nn.pool(x, (self.pool_size,), strides=(1,),
                       padding='SAME', pooling_type='MAX', data_format='NWC')

        x = K.expand_dims(x, axis=-2)
        x = K.depthwise_conv2d(x, self.blur_kernel, padding='same', strides=(self.pool_size, self.pool_size))
        x = K.squeeze(x, axis=-2)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.ceil(input_shape[1] / 2)), input_shape[2]






class AverageBlurPooling1D(Layer):

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(AverageBlurPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 2:
            bk = np.array([1, 1])
        elif self.kernel_size == 3:
            bk = np.array([1, 2, 1])
        elif self.kernel_size == 5:
            bk = np.array([1, 4, 6, 4, 1])
        else:
            if (math.log(self.kernel_size - 1, 2) / math.ceil(math.log(self.kernel_size - 1, 2))) != 1:
                raise ValueError
            else:
                blur_kernel = np.array([1, 1])
                for i in range(int(math.log(self.kernel_size - 1, 2))):
                    blur_kernel = np.convolve(blur_kernel, blur_kernel)

        bk = bk / np.sum(bk)
        bk = np.repeat(bk, input_shape[2])
        bk = np.reshape(bk, (self.kernel_size, 1, input_shape[2], 1))
        blur_init = keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, 1, input_shape[2], 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(AverageBlurPooling1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        x = tf.nn.pool(x, (self.pool_size,), strides=(1,), padding='SAME', pooling_type='AVG',
                       data_format='NWC')
        x = K.expand_dims(x, axis=-2)
        x = K.depthwise_conv2d(x, self.blur_kernel, padding='same', strides=(self.pool_size, self.pool_size))
        x = K.squeeze(x, axis=-2)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.ceil(input_shape[1] / 2)), input_shape[2]




class BlurPool1D(Layer):

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(BlurPool1D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 2:
            bk = np.array([1, 1])
        elif self.kernel_size == 3:
            bk = np.array([1, 2, 1])
        elif self.kernel_size == 5:
            bk = np.array([1, 4, 6, 4, 1])
        else:
            if (math.log(self.kernel_size - 1, 2) / math.ceil(math.log(self.kernel_size - 1, 2))) != 1:
                raise ValueError
            else:
                blur_kernel = np.array([1, 1])
                for i in range(int(math.log(self.kernel_size - 1, 2))):
                    blur_kernel = np.convolve(blur_kernel, blur_kernel)

        bk = bk / np.sum(bk)
        bk = np.repeat(bk, input_shape[2])
        bk = np.reshape(bk, (self.kernel_size, 1, input_shape[2], 1))
        blur_init = keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, 1, input_shape[2], 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(BlurPool1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        x = K.expand_dims(x, axis=-2)
        x = K.depthwise_conv2d(x, self.blur_kernel, padding='same', strides=(self.pool_size, self.pool_size))
        x = K.squeeze(x, axis=-2)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.ceil(input_shape[1] / 2)), input_shape[2]

class GlobalWeightedAveragePooling1D(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, **kwargs)
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], 1),
                                      initializer='ones',
                                      trainable=True)
        Layer.build(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2],

    def call(self, x):

        x = x*self.kernel
        x = K.mean(x, axis=1)

        return x


class GlobalWeightedMaxPooling1D(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, **kwargs)
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], 1),
                                      initializer='ones',
                                      trainable=True)
        Layer.build(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2],

    def call(self, x):

        x = x*self.kernel
        x = K.max(x, axis=1)

        return x