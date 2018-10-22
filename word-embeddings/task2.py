import scipy.io as sio
import numpy as np
import sys
sys.path.append('/home/aranguri/Desktop/dev/nns/rnn')
from utils import *

class Task:
    def __init__(self, batch_size):
        data = sio.loadmat('datasets/data.mat')
        test, train, val, words = data['data'][0][0]
        self.words = np.concatenate(([None], words[0]))

        self.vocab_size = len(self.words)
        self.num_words = 3

        train = train[:, :-50]
        train = train.reshape(train.shape[0], batch_size, -1)
        train = train.swapaxes(0, 2).swapaxes(1, 2)

        self.train_xs, self.train_ts = train[:, 0:3], train[:, 3]
        val_xs, self.val_ts = val[0:3], val[3]
        val_xs = np.array([[one_of_k(n, self.vocab_size) for n in inpt] for inpt in val_xs])
        self.val_xs = val_xs.swapaxes(1, 2)

    def one_of_k(self, x):
        x = np.array([[one_of_k(case, self.vocab_size) for case in letter] for letter in x])
        x = x.swapaxes(1, 2)
        return x

    def val_data(self):
        return self.val_xs[:, :, :1000], self.val_ts[:1000]

    def test_data(self):
        return self.test_xs, self.test_ts
