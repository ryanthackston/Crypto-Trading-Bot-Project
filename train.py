import os
import random

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2 as l2r
import numpy as np
import pandas as pd
from model import Res_Cryo
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import to_categorical
from fitonecycle import OneCycleLR
from cfg import *
from tensorflow.python.client import device_lib
from generator import CryptoSequence

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def main(session: str,
         optim: int = 1,
         L: int = 128,
         normalization: str = 'batch',
         v_normalization: str = 'global',
         workers: int = 1,
         l2: float = 0.01,
         kernel_initializer: str = 'glorot_uniform',
         random_seed: int = None):
    model_write_path = 'weights2/%s/' %session
    os.makedirs(model_write_path, exist_ok=True)
    print(get_available_gpus())
    # [BxLXF]
    #X = np.random.rand(32, L, 17) # note to self when loading into keras, its [Batch_Size, number of inputs, channel_size
    #Y = np.random.randint(0, 2, size=(32, 3))
    train_df = pd.read_csv('data3/train.csv', parse_dates=True)
    valid_df = pd.read_csv('data3/valid.csv', parse_dates=True)
    test_df = pd.read_csv('data3/test.csv', parse_dates=True)
    d = dict(zip(LABEL, range(0, 3)))
    train_df['Encode'] = train_df[y_label[0]].map(d, na_action='ignore')
    valid_df['Encode'] = valid_df[y_label[0]].map(d, na_action='ignore')
    test_df['Encode'] = test_df[y_label[0]].map(d, na_action='ignore')
    x_train, y_train = train_df.loc[:, x_label].to_numpy(), train_df[y_label[0]].to_numpy()
    x_valid, y_valid = valid_df.loc[:, x_label].to_numpy(), valid_df[y_label[0]].to_numpy()
    x_test, y_test = test_df.loc[:, x_label].to_numpy(), test_df[y_label[0]].to_numpy()

    label_encoder = LabelEncoder().fit(LABEL)
    # y_train = label_encoder.transform(y_train)
    # y_valid = label_encoder.transform(y_valid)
    # y_test = label_encoder.transform(y_test)
    oneshot_label_encoder = OneHotEncoder(sparse=False)
    y_train = oneshot_label_encoder.fit_transform(train_df[[y_label[0]]])
    y_valid = oneshot_label_encoder.fit_transform(valid_df[[y_label[0]]])
    y_test = oneshot_label_encoder.fit_transform(test_df[[y_label[0]]])
    train_seq = CryptoSequence(X=x_train, y=y_train, stage='train', batch_size=BS, normalization=normalization)
    valid_seq = CryptoSequence(X=x_valid, y=y_valid, batch_size=BS, normalization=normalization)
    test_seq = CryptoSequence(X=x_test, y=y_test, stage='test', batch_size=BS, normalization=normalization)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    if l2 is not None:
        kernel_regularizer = l2r(l2)
    else:
        kernel_regularizer = None

    model, loss_fns, metrics = Res_Cryo(dims=dims,
                                        L=L,
                                        pooling=pooling,
                                        kernel_regularizer=kernel_regularizer,
                                        kernel_initializer=kernel_initializer)
    class_weight = dict(enumerate(class_weights2))
    if optim ==1:
        radam = tfa.optimizers.RectifiedAdam(learning_rate=lr_range[1],
                                             beta_1=betas[0],
                                             beta_2=betas[1],
                                             epsilon=eps,
                                             weight_decay=wd,
                                             min_lr=lr_range[0])
        optimizer = tfa.optimizers.Lookahead(radam, sync_period=k, slow_step_size=alpha)
        filename = model_write_path + "%s_epoch-{epoch:02d}_val_loss-{val_loss:.4f}.h5" % session
        check_point = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True,
                                      save_weights_only=True, mode='min')
        callbacks = [check_point, reduce_lr]
        model.compile(optimizer=optimizer, loss=loss_fns, metrics=metrics)
        model.fit_generator(train_seq, validation_data=valid_seq, epochs=epochs, callbacks=callbacks, workers=workers, class_weight=class_weight)
        #inference
        # pred = model.predict_generator(test_seq, verbose=1)
        # cl = np.round(pred)
        # test_df['Predict'] = cl
        # test_df.to_csv(session + 'result.csv', index=False)

    else:
        optimizer = SGD(lr=1e-5, momentum=0.9)
        clr_triangle = OneCycleLR(L, BS, lr_range[1], end_percentage=0.1, scale_percentage=0.2)
        filename = model_write_path +"%s_epoch-{epoch:02d}_val_loss-{val_loss:.4f}.h5" % session
        check_point = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True,
                                      save_weights_only=True, mode='min')
        callbacks = [clr_triangle, check_point, reduce_lr]
        model.compile(optimizer=optimizer, loss=loss_fns, metrics=metrics, )
        model.fit_generator(train_seq, validation_data=valid_seq, epochs=epochs, callbacks=callbacks, workers=workers, shuffle=True, class_weight=class_weight)
        #inference
        # pred = model.predict_generator(test_seq, verbose=1)
        # cl = np.round(pred)
        # test_df['Predict'] = cl
        # test_df.to_csv(session + 'result.csv', index=False)


if __name__=="__main__":
    '''
    Change Blur = " " for any of the blur pool method [ blur, avg, or max]
    Change optim in function call to 1 or 0 where [ Ranger = 1, FitoneCycle = 0]
    '''
    main(session='FitOne_avg_W_SE_256', optim=0, L=256)
