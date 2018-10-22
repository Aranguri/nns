import sys
sys.path.append('../')
from utils import *

class Task:
    def __init__(self, seq_length, batch_size):
        self.seq_length = seq_length
        self.batch_size = batch_size
        text = open('../datasets/pg.txt').read()#[:1000000]
        text = clean_text(text)
        self.vocab_size, self.word_to_i, self.i_to_word, self.data = tokenize_text_simple(text)
        self.data = self.data[:-47].reshape(-1, batch_size)
        self.i = 0

    def next_batch(self):
        batch = self.data[self.i:self.i + self.seq_length]
        x = [one_of_k(x, self.vocab_size) for x in batch]
        x = np.array(x).swapaxes(1, 2)
        t = self.data[self.i + 1:self.i + self.seq_length + 1]
        self.i += self.seq_length
        self.i = 0 if self.i > len(self.data) else self.i
        return x, t

    def ixs_to_words(self, ixs):
        return self.i_to_word[ixs]
