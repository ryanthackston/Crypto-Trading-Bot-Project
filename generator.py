import numpy as np
from tensorflow.keras.utils import Sequence
import random
from cfg import *
from enum import Enum


class Normalization(Enum):
    NONE = 0
    BATCH = 1
    BATCH_F = 2
    SAMPLE = 3
    SAMPLE_F = 4
    GLOBAL = 5

    @staticmethod
    def from_str(s):
        if s == 'batch':
            return Normalization.BATCH
        elif s == 'batch_f':
            return Normalization.BATCH_F
        elif s == 'global':
            return Normalization.GLOBAL
        elif s == 'sample':
            return Normalization.SAMPLE
        elif s == 'sample_f':
            return Normalization.SAMPLE_F
        else:
            return Normalization.NONE


class CryptoSequence(Sequence):

    def __init__(self,
                 X,
                 y,
                 stage: str = "train",
                 batch_size: int = 128,
                 normalization=Normalization.BATCH,
                 augment: bool = False,
                 ):

        self.batch_size = batch_size
        self.stage = stage
        self.X = X
        self.y = y
        self.class_weights = class_weights2
        self.normalization = Normalization.from_str(normalization)
        self.augment = augment
        #self.sample_map = [i for i in range(0, int(np.ceil(len(self.X)/self.batch_size)))]
        self.sample_map = [i for i in range(0, len(self) * self.batch_size)]
        self.epochs = 0
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        u = np.mean(batch_x, axis=0)
        o = np.std(batch_x, axis=0)
        batch_x -= u
        batch_x /= o
        batch_x = batch_x[..., np.newaxis]
        return batch_x, batch_y
