import sys
sys.path.append('../')
from utils import *

class Task:
    def __init__(self, seq_length):
        self.seq_length = seq_length
        text = open('../datasets/shakespeare.txt').read()
        text = clean_text(text)
        self.vocab_size, word_to_i, self.data = tokenize_text(text)
        self.i = 0

    def next_batch(self):
        x = self.data[self.i:self.i + self.seq_length]
        x = one_of_k(x, self.vocab_size).reshape(self.seq_length, -1, 1)
        t = self.data[self.i + 1:self.i + self.seq_length + 1]
        t = np.array(t).reshape(-1, 1)
        self.i += 1
        return x, t
