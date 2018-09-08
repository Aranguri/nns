'''
I will try k nearest neighbors with texts. I will take 10 different sentences, each one is a category.
I will create training and test cases changing some characters from the original sentence.
Question: in cs231n, I read that there is a 'model' sentence with which each sentence is compared. Maybe we can reconstruct the text in this way
'''
import random
import string
import numpy as np
from pprint import pprint

class SameText:
    train = 200
    test = 5
    level = .1
    k = 5

    def __init__(self):
        self.sens = open('tasks/texts/same_text/meditations.txt', 'r').read()
        self.sens = self.sens.split('\n')[:-1]
        self.train_x = [self.mutate(sen) for sen in self.sens for _ in range(self.train)]
        self.train_y = np.array([i for i in range(10) for _ in range(self.train)])

        self.test_x = [self.mutate(sen) for sen in self.sens for _ in range(self.test)]
        self.test_y = [i for i in range(10) for _ in range(self.test)]

    def mutate(self, sen):
        sen = list(sen)
        for i in range(len(sen)):
            if random.uniform(0, 1) > self.level:
                sen[i] = random.choice(string.ascii_lowercase)
        return ''.join(sen)

    def main(self):
        results = []
        for x, y in zip(self.test_x, self.test_y):
            dists = [self.distance(x, other_x) for other_x in self.train_x]
            ind = np.argpartition(dists, self.k)[:self.k]
            counts = np.bincount(ind)
            results.append(y == self.train_y[np.argmax(counts)])
            #self.print_state(x, y, dists)
        return np.average(results)

    def print_state(self, x, y, dists):
        print ('Category {}'.format(y))
        print (x)
        print (self.train_x[np.argmin(dists)])
        print (y == self.train_y[np.argmin(dists)])
        print (np.array(dists).reshape(10, self.train))
        print ('\n\n')

    @staticmethod
    def distance(x1, x2):
        # return sum([(ord(c1) - ord(c2)) ** 2 for c1, c2 in zip(x1, x2)]) # 0.67, 0.7, 0.75, 0.73
        # return sum([abs(ord(c1) - ord(c2)) for c1, c2 in zip(x1, x2)]) #0.78, 0.83, 0.7, 0.77
        return 1 / (sum([c1 == c2 for c1, c2 in zip(x1, x2)]) + .000001) #.95, .94, .92

results = [SameText().main() for _ in range(3)]
print (np.average(results), end=' ')
print (results)
