''' Numpy implementation of a CNN with the option for conv and pool layers '''

import numpy as np
from utils import *
from tasks.mnist_task import Task
import time

class ConvNN:
    def __init__(self):
        self.learning_rate = 1e-2
        self.epochs = 10000
        self.shapes = 1 * [[('conv', 5)]]
        task = Task()
        self.input_size, self.output_size = task.sizes()
        self.Ws = [{i: np.random.randn(1, m) if type == 'conv' else None
                    for i, (type, m) in enumerate(s)} for s in self.shapes]
        self.final_Ws = [np.random.randn(self.output_size, end_size(s, self.input_size))
                         for s in self.shapes]
        self.train_x, self.train_y = task.train_x, task.train_y

    def run(self):
        for i in range(self.epochs):
            costs, accuracies = [], []
            for x, y in zip(self.train_x, self.train_y):
                self.y = y
                self.forward(expand(x))
                self.backward()
                costs.append(self.compute_cost()[0])
                accuracies.append(self.compute_acc())
            print ('Epoch {}: Cost {}. Accuracy: {}'.format(i, np.mean(costs), np.mean(accuracies)))

    def forward(self, inpt):
        self.Xs, self.Zs = [], []
        self.final_Z = np.zeros((self.output_size, 1))

        for i, shape in enumerate(self.shapes):
            self.Xs.append({0: inpt})
            self.Zs.append({})
            for j, (type, size) in enumerate(shape):
                x = self.Xs[i][j]
                if type == 'pool':
                    self.Xs[i][j + 1] = np.array([max(x[k:k + size]) for k in range(0, len(x) - size + 1, size)])
                elif type == 'conv':
                    self.Zs[i][j + 1] = np.array([np.dot(self.Ws[i][j], x[k:k + size])[0] for k in range(len(self.Xs[i][j]) - size + 1)])
                    self.Xs[i][j + 1] = sigmoid(self.Zs[i][j + 1])
            self.final_Z += np.dot(self.final_Ws[i], self.Xs[i][len(shape)])

        self.target = sigmoid(self.final_Z)

    def backward(self):
        dCdZ = (self.target - self.y) * sigmoid_prime(self.final_Z)
        dCdW = dCdZ.dot(self.Xs[0][1].T)
        self.final_Ws[0] -= self.learning_rate * dCdW

        dCdZ = np.dot(self.final_Ws[0].T, dCdZ) * sigmoid_prime(self.Zs[0][1])
        dCdW = np.dot(dCdZ, self.Xs[0][0].T)
        dCdW = conv_mask(dCdW)
        self.Ws[0][0] -= self.learning_rate * dCdW

    def compute_cost(self):
        return (self.y - self.target) ** 2 / 2

    def compute_acc(self):
        return (self.target > .5) == self.y

conv_nn = ConvNN()
conv_nn.run()

'''
There is saturation. That's the problem.
How to continue:
* Design a very simple task where convolutional nets make sense.
* Change sigmoid to relu
* Add biases
* Change cost function to cross-entropy xd
* Add pool layers
* Make possible more than one layer in a given shape
* Make possible more than one shape in a given shapes
* Design a more complex task
'''
