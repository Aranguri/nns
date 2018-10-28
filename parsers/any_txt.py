import sys
sys.path.append('../')
from utils import *

class Task:
    def __init__(self, seq_length, batch_size):
        self.seq_length = seq_length
        self.batch_size = batch_size
        text = open('../datasets/pg.txt').read()[:1000000]
        words = clean_text(text)
        self.vocab_size, self.word_to_i, self.i_to_word, self.data = tokenize_words_simple(words)
        self.data = self.data[:-(seq_length + len(self.data) % seq_length)]
        self.data = self.data.reshape(-1, seq_length)
        self.train = self.data[:-1000]
        self.dev = self.data[-1000:]
        self.t_i = 0
        self.d_i = 0

    def train_batch(self):
        x = self.train[self.t_i:self.t_i  + self.batch_size]
        batch_t = self.train[self.t_i + 1:self.t_i + self.batch_size + 1]
        t = [one_of_k(t, self.vocab_size) for t in batch_t]

        self.t_i += self.batch_size
        self.t_i = 0 if self.t_i + self.batch_size >= len(self.train) else self.t_i
        return x, t

    def dev_batch(self):
        x = self.dev[self.d_i:self.d_i + self.batch_size]
        batch_t = self.dev[self.d_i + 1:self.d_i + self.batch_size + 1]
        t = [one_of_k(t, self.vocab_size) for t in batch_t]

        self.d_i += self.batch_size
        self.d_i = 0 if self.d_i + self.batch_size >= len(self.dev) else self.d_i
        return x, t

    def ixs_to_words(self, ixs):
        return self.i_to_word[ixs]
