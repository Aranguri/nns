import random
import numpy as np
import sys
sys.path.append('/home/aranguri/Desktop/dev/nns/rnn')
from utils import *

class Task:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        cases = restore('processed_shakespeare')
        xs, ts = zip(*cases[0][:10])
        unique_words = set([word for case in xs for word in case])
        word_to_i = {word: i for i, word in enumerate(unique_words)}
        self.vocab_size = len(unique_words)
        xs = [[one_of_k(word_to_i[word], self.vocab_size) for word in words] for words in xs]
        self.data = list(zip(xs, ts))[:9]
        self.val_xs, self.val_ts = zip(*list(zip(xs, ts))[9:])
        self.val_xs, self.val_ts = np.array(self.val_xs), np.array(self.val_ts)
        self.val_xs = self.val_xs.swapaxes(0, 1).swapaxes(1, 2)
        self.n = -1

    def next_batch(self):
        if self.n == len(self.data) // self.batch_size or self.n == -1:
            self.n = 0
            np.random.shuffle(self.data)
            self.xs, self.ts = zip(*self.data)
            self.xs, self.ts = np.array(self.xs), np.array(self.ts)

        ixs = random.sample(range(len(self.xs)), self.batch_size)
        self.n += 1
        xs = self.xs[ixs].swapaxes(0, 1).swapaxes(1, 2)
        to_return = xs, self.ts[ixs]
        self.xs = np.delete(self.xs, ixs, 0)
        self.ts = np.delete(self.ts, ixs, 0)
        return to_return

    def val_data(self):
        return self.val_xs, self.val_ts
