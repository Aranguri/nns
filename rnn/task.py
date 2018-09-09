import numpy as np
from utils import psh, expand

class Task:
    def __init__(self, seq_length, batch_size):
        all_data = open('/home/aranguri/Desktop/dev/nns/rnn/datasets/input.txt', 'r').read()
        self.chars = list(set(all_data))
        self.vocab_size = len(self.chars)
        self.char_to_i = {char: i for i, char in enumerate(self.chars)}
        self.i_to_char = np.array(self.chars)
        self.seq_length = seq_length
        self.n = 0

        all_data = np.array([self.one_hot(char) for char in all_data])
        tr_data, self.val_data = all_data[:1000000], all_data[1000000:]
        if len(tr_data) % batch_size != 0:
            tr_data = tr_data[:-(len(tr_data) % batch_size)]
        tr_data = tr_data.reshape(batch_size, -1, self.vocab_size).T
        self.tr_data = np.transpose(tr_data, (1, 0, 2))

    def next_batch(self):
        n, sl = self.n, self.seq_length
        xs = self.tr_data[n:n + sl]
        ys = np.argmax(self.tr_data[n + 1:n + sl + 1], 1)
        self.n = (n + sl) % (len(self.tr_data) - 2 * sl)
        return xs, ys

    def get_val_data(self):
        start = np.random.randnint(len(self.val_data) - 1000)
        return self.val_data[start:start + 1000]

    def one_hot(self, char=None, num=None):
        array = np.zeros((self.vocab_size))
        if char != None:
            array[self.char_to_i[char]] = 1
        elif num != None:
            array[num] = 1
        return array

    def array_to_sen(self, array):
        return ''.join(self.i_to_char[array])

    def rand_x(self):
        return expand(self.one_hot(np.random.choice(self.chars)))
