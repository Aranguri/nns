import random
import numpy as np
import sys
sys.path.append('/home/aranguri/Desktop/dev/nns/rnn')
from utils import *

class Task:
    def __init__(self, batch_size):
        tr_size = 9000
        val_size = 100
        self.batch_size = batch_size
        cases = restore('processed_shakespeare')
        xs, ts = zip(*cases[0][:tr_size + val_size])
        unique_words = sorted(set([word for case in xs for word in case]))
        self.word_to_i = {word: i for i, word in enumerate(unique_words)}
        self.vocab_size = len(unique_words)
        self.data = list(zip(xs, ts))[:tr_size]

        val_xs, self.val_ts = np.array(xs[tr_size:]), np.array(ts[tr_size:])
        val_xs = [[one_of_k(self.word_to_i[word], self.vocab_size) for word in words] for words in val_xs]
        self.val_xs = np.array(val_xs).swapaxes(0, 1).swapaxes(1, 2)

    def next_batch(self):
        if not hasattr(self, 'n') or self.n == len(self.data) // self.batch_size:
            self.n = 0
            np.random.shuffle(self.data)
            self.xs, self.ts = zip(*self.data)
            self.xs, self.ts = np.array(self.xs), np.array(self.ts)

        ixs = random.sample(range(len(self.xs)), self.batch_size)
        self.n += 1
        xs = [[one_of_k(self.word_to_i[word], self.vocab_size) for word in words] for words in self.xs[ixs]]
        xs = np.array(xs).swapaxes(0, 1).swapaxes(1, 2)
        to_return = xs, self.ts[ixs]
        self.xs = np.delete(self.xs, ixs, 0)
        self.ts = np.delete(self.ts, ixs, 0)
        return to_return

    def val_data(self):
        return self.val_xs, self.val_ts
