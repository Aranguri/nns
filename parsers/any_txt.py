import sys
sys.path.append('../')
from utils import *

class Task:
    def __init__(self, seq_length, batch_size):
        self.seq_length = seq_length
        self.batch_size = batch_size
        text = open('../datasets/shakespeare.txt').read()[:1000000]
        words = clean_text(text)
        self.vocab_size, self.word_to_i, self.i_to_word, self.data = tokenize_words_simple(words)
        self.data = self.data[:-(batch_size + len(self.data) % batch_size)]
        self.data = self.data.reshape(-1, batch_size)
        self.i = 0

    def next_batch(self):
        batch = self.data[self.i:self.i + self.seq_length]
        #x = [one_of_k(x, self.vocab_size) for x in batch]
        x = np.array(batch).swapaxes(0, 1).astype('int32')
        batch_t = self.data[self.i + 1:self.i + self.seq_length + 1]
        t = [one_of_k(t, self.vocab_size) for t in batch_t]
        t = np.array(t).swapaxes(0, 1)
        self.i += self.seq_length
        self.i = 0 if self.i + self.seq_length > len(self.data) else self.i
        return x, t

    def ixs_to_words(self, ixs):
        return self.i_to_word[ixs]
