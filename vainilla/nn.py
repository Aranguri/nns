import numpy as np
import os
from copy import deepcopy
from gen_text_task import Task
from utils import sigmoid, sigmoid_prime, string_to_bits, normalize, bits_to_string, psh, mask_like

class NN():
    exp_name = '4-char-6-words-win-2-v2'

    def __init__(self):
        self.task = Task()
        self.sizes = [len(self.task.train_x[0]), 100, 30, len(self.task.train_y[0])]
        self.learning_rate = .1
        self.eta = .00#1
        self.p_dropout = 0
        self.batch_size = 64
        self.epoch_size = 10000

        self.Ws = [np.random.randn(m, (n + 1)) for m, n in zip(self.sizes[1:], self.sizes)]
        self.costs = []
        self.accuracies = []
        #self.restore()

    def run(self):
        for i in range(self.epoch_size):
            try:
                self.costs.append([])
                self.accuracies.append([])
                self.train()
                self.validate()
                for _ in range(100): self.sample()
                print ('Epoch {}: Cost {}.'.format(i, np.average(self.costs[-1]), end=''))
                print ('Acc: {}'.format(np.average(self.accuracies[-1])))
            except KeyboardInterrupt:
                key = input()
                if key == 's' or key == 'b':
                    self.save()
                if key == 'p' or key == 'b':
                    plt.plot(accuracies)
                    plt.show(block=False)
                    time.sleep(10000)
        return self.accuracies[-1]

    def train(self):
        size = int(len(self.task.train_x) / self.batch_size) * self.batch_size
        indices = np.random.permutation(np.arange(size))
        indices = indices.reshape((int(len(self.task.train_x) / self.batch_size), self.batch_size))
        batches_x = [self.task.train_x[i] for i in indices]
        batches_y = [self.task.train_y[i] for i in indices]

        for batch_x, batch_y in zip(batches_x[0:10], batches_y[0:10]):
            self.batch_X = batch_x
            self.batch_Y = batch_y
            self.sgd()

    def validate(self):
        for x, y in zip(self.task.validation_x, self.task.validation_y):
            self.Xs = [x]
            self.forward_pass()
            #print (bits_to_string(x) + self.task.get_chars()[y])
            #print (bits_to_string(x) + self.task.get_chars()[np.argmax(self.Xs[-1])] + '\n')
            self.accuracies[-1].append(np.argmax(y) == np.argmax(self.Xs[-1]))

    def sample(self):
        sentence = self.task.train_x[np.random.randint(self.sizes[-1])]
        print (bits_to_string(sentence), end='|')

        for _ in range(1):
            self.Xs = [sentence]
            self.forward_pass()
            #char = np.random.choice(self.task.get_chars(), p=normalize(self.Xs[-1]))
            char = self.task.get_chars()[np.argmax(self.Xs[-1])]
            sentence = np.concatenate((sentence[8:], string_to_bits(char)))
            print (char, end='')
        print ('\n\n')

    def sgd(self):
        batch_dCdWs = []

        for X, Y in zip(self.batch_X, self.batch_Y):
            self.Xs = [X]
            self.Y = Y
            self.forward_pass()
            self.costs[-1].append(self.compute_cost())
            batch_dCdWs.append(self.backward_pass())

        dCdWs = self.learning_rate * np.average(batch_dCdWs, axis=0)
        self.Ws = [W - (dW + self.eta * W) for W, dW in zip(self.Ws, dCdWs)]

    def forward_pass(self):
        self.Zs = []
        for i, W in enumerate(self.Ws):
            self.Xs[i] = deepcopy(self.Xs[i] * mask_like(self.Xs[i], self.p_dropout))
            self.Xs[i] = np.append(self.Xs[i], [[1]], axis=0) #bias term
            self.Zs.append(np.dot(W, self.Xs[i]))
            self.Xs.append(sigmoid(self.Zs[i]))

    def backward_pass(self):
        dCdZ = ((self.Xs[-1] - self.Y) * sigmoid_prime(self.Zs[-1])) # 1x10 | 10x1
        dCdWs = [np.dot(dCdZ, self.Xs[-2].T)] # dCdZ * dZdWs = 1x10 · 10x10x30 = 1x10x30 | 10x1 · 1x30 = 10x30

        for i in range(len(self.sizes) - 2, 0, -1):
            sp = sigmoid_prime(self.Zs[i - 1])
            dCdZ = np.dot(self.Ws[i][:,:-1].T, dCdZ) * sp       # dCdZ * dZdZ = 1x10 · 10·30 = 1x30 | (30x10 · 10x1) * 30x1 = 30x1
            dCdWs.append(np.dot(dCdZ, self.Xs[i - 1].T)) # dCdZ * dZdW = 1x30 · 30x30x100 = 1x30x100 | 30x1 · 1x100 = 30x100

        return dCdWs[::-1]

    def compute_cost(self):
        return sum((self.Y - self.Xs[-1]) ** 2) / 2 + self.eta * np.average([np.average(W ** 2) for W in self.Ws])

    def save(self):
        path = 'savings/{}'.format(self.exp_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, sizes=self.sizes, Ws=self.Ws)

    def restore(self):
        npzfile = np.load('savings/{}.npz'.format(self.exp_name))
        self.sizes, self.Ws = [item for _, item in npzfile.items()]
        print ('Net restored')

NN().run()
