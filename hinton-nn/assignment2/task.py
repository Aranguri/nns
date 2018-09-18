import sys
sys.path.append('/home/aranguri/Desktop/dev/nns/rnn')
from utils import *
import scipy.io as sio
import numpy as np

class Task:
    batch_size = 100

    def __init__(self):
        data = sio.loadmat('data.mat')
        test, self.train, val, self.words = data['data'][0][0] #Maybe test and validation are inverted.
        self.val_xs, self.val_ts = val[0:3], val[3]
        self.test_xs, self.test_ts = test[0:3], test[3]
        self.vocab_size = len(self.words[0]) + 1
        self.n = 0

    def next_batch(self):
        if self.n + self.batch_size >= self.train.shape[1]:
            self.n = 0
            np.random.shuffle(self.train)
        end = self.n + self.batch_size
        train_xs = self.train.T[self.n:end, 0:3]
        train_xs = np.array([one_of_k(inpt, self.vocab_size) for inpt in train_xs]).T
        train_ts = self.train[3, self.n:end]
        self.n = end
        return train_xs, train_ts

    def val_test(self):
        return self.val_xs, self.val_ts, self.test_xs, self.test_ts, self.words
