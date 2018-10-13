import sys
sys.path.append('../')
from utils import *
import numpy as np
import random

class Task:
    def __init__(self, seq_length, batch_size):
        self.seq_length = seq_length
        self.docs = restore('processed-wiki')[0]
        self.word_to_i = {}

    def next_batch(self):
        same = np.random.randint(2)
        p1, p2 = None, None

        if same:
            doc = random.sample(self.docs, 1)[0]
            p1, p2 = random.sample(doc, 2)
        else:
            doc1, doc2 = random.sample(self.docs, 2)
            p1 = random.sample(doc1, 1)[0]
            p2 = random.sample(doc2, 1)[0]
        (x1, new_w1), (x2, new_w2) = self.encode(p1), self.encode(p2)
        x1, x2 = one_of_k(x1, len(self.word_to_i)), one_of_k(x2, len(self.word_to_i))
        x1, x2 = x1.reshape(*x1.shape, -1), x2.reshape(*x2.shape, -1)
        #return x1, x2, new_w1 + new_w2, same
        return x1, 'was' in x1

    def encode(self, text):
        text = clean_text(text)
        x, self.word_to_i, new_words = tokenize_text(text, self.word_to_i)
        x = x[:self.seq_length]
        return x, new_words
