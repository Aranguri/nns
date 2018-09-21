import numpy as np
from utils import ps, expand

class Task:
    def __init__(self, seq_length, batch_size):
        all_data = open('/home/aranguri/Desktop/dev/nns/rnn/datasets/input_tf.txt', 'r').read()
        self.chars = list(sorted(set(all_data)))
        self.vocab_size = len(self.chars)
        self.char_to_i = {char: i for i, char in enumerate(self.chars)}
        self.i_to_char = np.array(self.chars)
        self.seq_length = seq_length
        self.n = 0

        all_data = np.array([self.one_of_k(char) for char in all_data])
        tr_data, self.val_data = all_data[:300000], all_data[300000:310000]
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
        start = np.random.randint(len(self.val_data) - 1000)
        xs = self.val_data[start:start + 1000]
        ys = self.val_data[start + 1:start + 1001]
        return xs, ys

    def one_of_k(self, char=None, num=None):
        array = np.zeros((self.vocab_size))
        if char != None:
            array[self.char_to_i[char]] = 1
        elif num != None:
            array[num] = 1
        return array

    def array_to_sen(self, array):
        return ''.join(self.i_to_char[array])

    def rand_x(self):
        return expand(self.one_of_k(np.random.choice(self.chars)))
