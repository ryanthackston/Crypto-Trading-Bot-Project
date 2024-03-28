import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from blurpool_1d import *
from cfg import *

class Squeeze_Excitation(object):
    def __init__(self, dims: int, ratio: int, name: str):
        self.name = name
        self.input = input
        self.channels = dims
        self.ratio = ratio
        self.pool = GlobalWeightedAveragePooling1D(name='%s_gwap' % self.name)
        self.dense1 = Dense(self.channels // self.ratio, activation='swish', name='%s_dense_swish' % self.name, use_bias=False)
        self.dense2 = Dense(self.channels, activation='sigmoid', name='%s_dense_soft' % self.name, use_bias=False)
        self.squeeze = Multiply(name='%s_excitation' % self.name)
    def __call__(self, x):
        # Squeeze
        x_i = x
        x = self.pool(x)
        # Excitation
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.squeeze([x_i, x])
        return x


class Res_Block(object):

    def __init__(self, shape: int, dims: int, name: str, pooling=None, kernel_regularizer=None, kernel_initializer: str = 'glorot_uniform'):
        self.name = name
        self.pooling = pooling
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.dims = dims
        self.bn_1 = BatchNormalization(name='%s_bn_1' % self.name)
        self.act_1 = Activation('swish', name='%s_act_1' % self.name)
        self.bn_2 = BatchNormalization(name='%s_bn_2' % self.name)
        self.act_2 = Activation('swish', name='%s_act_2' % self.name)

        self.conv_1 = Conv1D(dims, shape,
                             name='%s_conv_1' % self.name,
                             use_bias=False,
                             padding='same',
                             kernel_regularizer=kernel_regularizer,
                             kernel_initializer=kernel_initializer)

        self.conv_2 = Conv1D(dims, shape,
                             name='%s_conv_2' % self.name,
                             strides=1 if pooling != 'None' else 2,
                             use_bias=True,
                             padding='same',
                             kernel_regularizer=kernel_regularizer,
                             kernel_initializer=kernel_initializer)

        if pooling == 'blur':
            self.pool = BlurPool1D(name='%s_Blurpool' % self.name)
        elif pooling == 'avg':
            self.pool = AverageBlurPooling1D(name='%s_AvgBlurpool' % self.name)
        elif pooling == 'max':
            self.pool = MaxBlurPooling1D(name='%s_MaxBlurpool' % self.name)
        else:
            self.pool = MaxPooling1D(name='%s_skip_pool' % self.name)

        if squeeze:
            self.squeeze = Squeeze_Excitation(dims=dims, ratio=1, name='%s_SE_block' % self.name)

        self.residual = Add(name='%s_residual' % self.name)

        self.skip_layer = Conv1D(self.dims, 1,
                                 use_bias=True,
                                 name='%s_skip_project' % self.name,
                                 padding='same',
                                 kernel_initializer=self.kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer)

    def __call__(self, x):
        xn = x
        xn = self.bn_1(xn)
        xn = self.act_1(xn)

        xa = xn

        xn = self.conv_1(xn)
        xn = self.bn_2(xn)
        xn = self.act_2(xn)
        xn = self.conv_2(xn)
        if squeeze:
            xn = self.squeeze(xn)


        if x.shape[-1] != xn.shape[-1]:
            xs = self.skip_layer(xa)
        else:
            xs = x


        y = self.residual([xn, xs])
        if self.pooling is not None:
            y = self.pool(y)

        return y

def Res_Cryo(dims: int = 32,
                   pooling='blur',
                   kernel_regularizer=None,
                   L: int= 1000,
                   n_feat: int = len(x_label),
                   kernel_initializer: str = 'glorot_uniform'):
    layer_input = Input((n_feat, 1))
    def cce(y_true, y_pred):
        return K.mean(K.categorical_crossentropy(y_true, y_pred))

    metrics = {'cls': ['acc', cce]}
    loss_fns = ['categorical_crossentropy']
    x = layer_input
    conv1 = Conv1D(dims, 5,
                   strides=1 if pooling != 'avg' else 2,
                   padding='same',
                   name='0_conv_1',
                   kernel_regularizer=kernel_regularizer,
                   kernel_initializer=kernel_initializer)
    bn1 = BatchNormalization(name='0_bn')
    act1 = ReLU(name='0_relu')

    if pooling == 'blur':
        pool = BlurPool1D(name='0_Blurpool')
    elif pooling == 'avg':
        pool = AverageBlurPooling1D(name='0_AvgBlurpool')
    elif pooling == 'max':
        pool = MaxBlurPooling1D(name='0_MaxBlurpool')
    else:
        pool = MaxPooling1D(name='%0_skip_pool')

    conv2 = Conv1D(dims, 3,
                   padding='same',
                   name='0_conv_2',
                   kernel_regularizer=kernel_regularizer,
                   kernel_initializer=kernel_initializer)

    res_1 = Res_Block(shape=3,
                         dims=dims,
                         name='res_block1',
                         kernel_regularizer=kernel_regularizer,
                         kernel_initializer=kernel_initializer,
                         pooling=pooling)

    res_2 = Res_Block(shape=3,
                         dims=dims * 2,
                         name='res_block2',
                         kernel_regularizer=kernel_regularizer,
                         kernel_initializer=kernel_initializer,
                         )

    res_3 = Res_Block(shape=3,
                         dims=dims * 4,
                         name='res_block3',
                         kernel_regularizer=kernel_regularizer,
                         kernel_initializer=kernel_initializer,
                         )

    gmp = GlobalWeightedMaxPooling1D(name='gwmp')
    cls2 = Dense(3, activation='softmax', name='cls', use_bias=False) # work for swish, currently working for softmax


    # Forward Pass
    x = conv1(x)
    x = bn1(x)
    x = act1(x)
    x = pool(x)
    x = conv2(x)
    x = res_1(x)
    x = res_2(x)
    x = res_3(x)
    x = gmp(x)
    x = cls2(x)

    model = Model(inputs=layer_input, outputs=x)
    model.summary()
    return model, loss_fns, metrics
