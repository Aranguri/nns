import sys
sys.path.append('../')
from utils import *
import numpy as np
import random

class Task:
    def __init__(self, seq_length, batch_size):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.docs = restore('processed-wiki')[0]
        self.word_to_i = {}
        self.words = []
        self.acc = []

    # Task 1: Ask nn to tell whether two sentences are about the same topic or not
    # Task 2: Ask nn if a sentence has a specific letter in it
    def next_batch_old(self):
        same = np.random.randint(2)
        x1, label, new_words = [], '', 0
        first = True

        while first or same != label:
            first = False
            p1, p2 = None, None

            if same:
                doc = random.sample(self.docs, 1)[0]
                p1, p2 = random.sample(doc, 2)
            else:
                doc1, doc2 = random.sample(self.docs, 2)
                p1 = random.sample(doc1, 1)[0]
                p2 = random.sample(doc2, 1)[0]
            (x1, new_w1), (x2, new_w2) = self.encode(p1), self.encode(p2)
            new_words += new_w1 + new_w2
            label = 2 in x1
            x1, x2 = one_of_k(x1, len(self.word_to_i)), one_of_k(x2, len(self.word_to_i))
            x1, x2 = x1.reshape(*x1.shape, -1), x2.reshape(*x2.shape, -1)
            #return x1, x2, new_w1 + new_w2, same
        self.acc.append([label])
        # print (self.acc)
        # print (np.mean(self.acc))
        return x1, label, new_words

    #Task 3: ask nn to distinguish between valid and invalid paragraphs.
    # Options: instead of taking random words for the corrupted version
    # of the document, just permutate the words of a valid document.
    # Also, tune how random words are.
    def next_batch(self):
        ys = np.random.randint(0, 2, size=(self.batch_size))
        xs = [[]] * self.batch_size

        for i, y in enumerate(ys):
            while len(xs[i]) < self.seq_length:
                if y or len(self.words) == 0:
                    doc = random.sample(self.docs, 1)[0]
                    p = random.sample(doc, 1)[0]
                else:
                    doc = random.sample(self.docs, 1)[0]
                    p = random.sample(doc, 1)[0]
                    #Task3
                    #p = random.sample(self.words, min(100, len(self.words)))
                    #p = ' '.join(p)
                xs[i], new_words = self.encode(p)
                if not y: np.random.shuffle(xs[i])
                self.words += new_words

        xs = np.array([one_of_k(x, len(self.word_to_i)) for x in xs])
        xs = xs.swapaxes(0, 1).swapaxes(1, 2)
        #print (self.decode(np.array(xs)), ys[0])
        return xs, ys

    def test_human(self):
        for i in range(10):
            x, valid, _ = self.next_batch()
            print (valid)
            print (self.decode(x))
            print ('\n\n')

    def encode(self, text):
        words = clean_text(text)
        x, self.word_to_i, new_words = tokenize_words(words, self.word_to_i)
        x = x[:self.seq_length]
        return x, new_words

    def decode(self, x):
        i_to_word = list(self.word_to_i.keys())
        x = np.argmax(x, 1)
        return ' '.join([i_to_word[w[0]] for w in x])
