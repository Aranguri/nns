import random
import numpy as np
import sys
sys.path.append('/home/aranguri/Desktop/dev/nns/rnn')
from utils import *

class Generator:
    def __init__(self):
        data = open('/home/aranguri/Desktop/dev/nns/rnn/datasets/input.txt').read()
        for c in ['!', '?', '.', ',', ':', ';', '--', '\'']:
            data = data.replace(c, f' {c}')
        data = data.replace('--', '-- ')
        sens = data.split('\n')
        all_words = [s.split(' ') for s in sens]
        all_words = [w for w in all_words if len(w) >= 5]
        self.vocab = set()
        for words in all_words:
            for word in words:
                self.vocab.add(word)
        xs_1 = [w for ws in all_words for w in zip(*(ws[i:] for i in range(5)))]
        xs_0 = [self.make_wrong(w, i) for i, w in enumerate(random.choices(xs_1, k=len(xs_1)))]
        xs = np.concatenate((xs_1, xs_0))
        ys = np.concatenate((np.ones(len(xs_1)), np.zeros(len(xs_1))))
        cases = list(zip(xs, ys))
        random.shuffle(cases)
        save('processed_shakespeare', cases)
        print ('finished')

    def make_wrong(self, words, i):
        words = list(words)
        words[2] = np.random.choice(list(self.vocab))
        return words

Task()
