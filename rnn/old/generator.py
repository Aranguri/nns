import numpy as np
import time
import matplotlib.pyplot as plt
from utils import cost, tanh_prime, expand, psh, normalize

class RNN:
    h_size = 100
    learning_rate = 1e-1
    batch_size = 32
    regul = .5
    initial_h = expand(np.zeros(h_size))
    filename = 'input8.txt'

    def __init__(self):
        self.h = self.initial_h
        self.text = open(self.filename, 'r').read()[:-1]
        chars = set(self.text)
        self.x_size = len(chars)

        self.i_to_char = {i: char for i, char in enumerate(chars)}
        self.char_to_i = {char: i for i, char in enumerate(chars)}

        self.W_xh = np.random.randn(self.h_size, self.x_size) * .01
        self.W_hh = np.random.randn(self.h_size, self.h_size) * .01
        self.W_hy = np.random.randn(self.x_size, self.h_size) * .01
        self.B_h = np.zeros((self.h_size, 1))
        self.B_y = np.zeros((self.x_size, 1))

    def forward_pass(self):
        self.prev_h = self.h#np.copy(self.h)
        self.h = np.tanh(np.dot(self.W_xh, self.x) + np.dot(self.W_hh, self.h) + self.B_h)
        return np.tanh(np.dot(self.W_hy, self.h) + self.B_y)

    def backward_pass(self):
        dCdY = self.y - self.target
        dYdZ = np.diag(tanh_prime(np.dot(self.W_hy, self.h)).reshape(self.x_size,))
        dZdWhy = np.repeat(self.h.T, self.x_size, axis=0)
        dYdWhy = np.dot(dYdZ, dZdWhy)
        dCdWhy = dCdY * dYdWhy
        dCdBy = np.dot(dCdY.T, dYdZ)

        dZdHt = self.W_hy
        dYdHt = np.dot(dYdZ, dZdHt)
        dHtdZ2 = tanh_prime(np.dot(self.W_xh, self.x) + np.dot(self.W_hh, self.h))
        dHtdZ2 = np.diag(dHtdZ2.reshape(self.h_size,))
        dZ2dWxh = np.repeat(self.x.T, self.h_size, axis=0)
        dZ2dWhh = np.repeat(self.prev_h.T, self.h_size, axis=0)
        dYdZ2 = np.dot(dYdHt, dHtdZ2)
        dYdWxh = np.dot(dYdZ2, dZ2dWxh)
        dYdWhh = np.dot(dYdZ2, dZ2dWhh)
        dCdWxh = np.repeat(np.dot(dCdY.T, dYdWxh), self.h_size, axis=0)
        dCdWhh = np.repeat(np.dot(dCdY.T, dYdWhh), self.h_size, axis=0)
        dCdBh = np.dot(dCdY.T, dYdZ2)

        return [dCdWxh, dCdWhh, dCdWhy, dCdBh, dCdBy]

    def diagnose(self, i):
        if i % 100 == 0:
            print ('Iteration: {}. Cost: {}'.format(i, np.average(self.costs[-10:])))
            x = np.random.randint(self.x_size)
            for j in range(200):
                print (self.i_to_char[x], end='')
                self.x = expand(np.eye(self.x_size)[x])
                y = self.forward_pass()
                y = normalize(y.reshape(self.x_size,))
                x = np.argmax(y)
                #x = np.random.choice(np.arange(self.x_size), p=y)
            print ('\n')

    def train(self):
        self.costs = []
        gradients = []

        for i, (char, target) in enumerate(zip(self.text, self.text[1:])):
            self.x = expand(np.eye(self.x_size)[self.char_to_i[char]])
            self.target = expand(np.eye(self.x_size)[self.char_to_i[target]])

            self.y = self.forward_pass()
            gradients.append(self.backward_pass())

            if (i + 1) % self.batch_size == 0:
                dCdWxh, dCdWhh, dCdWhy, dCdBh, dCdBy = np.average(gradients, axis=0)
                self.W_xh -= self.learning_rate * dCdWxh
                self.W_hh -= self.learning_rate * dCdWhh
                self.W_hy -= self.learning_rate * dCdWhy
                self.B_h -= self.learning_rate * dCdBh.T
                self.B_y -= self.learning_rate * dCdBy.T

            self.costs.append(cost(self.target, self.y))
            self.diagnose(i)

        plt.plot(self.costs)
        plt.show(block=False)
        time.sleep(1000)

rnn = RNN()
rnn.train()
