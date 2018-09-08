import itertools
import numpy as np

class NN():
    learning_rate = .03
    regularization = .003
    batch_size = 128
    epoch_size = 50
    test_runnings = 10000

    def __init__(self):
        self.sizes = [784, 100, 30, 10]
        self.Ws = [np.random.randn(m, n) for m, n in zip(self.sizes[1:], self.sizes)]
        self.Bs = [np.random.randn(m, 1) for m in self.sizes[1:]]

    def run(self):
        accuracies = []

        for i in itertools.count():
            self.costs = []
            self.train()
            accuracies.append(self.test())
            print ('Epoch {}: Cost {}. Acc {}'.format(i, np.average(self.costs), accuracies[-1]))

    def train(self):
        for _ in range(self.epoch_size):
            indices = np.random.choice(self.train_images.shape[0], self.batch_size)
            self.batch_X = [self.train_images[i] for i in indices]
            self.batch_Y = [self.train_labels[i] for i in indices]
            self.sgd()

    def sgd(self):
        batch_dCdWs = []
        batch_dCdBs = []

        for X, Y in zip(self.batch_X, self.batch_Y):
            self.Xs = [X]
            self.Y = Y
            self.forward_pass()
            dCdWs, dCdBs = self.backward_pass()
            batch_dCdWs.append(dCdWs)
            batch_dCdBs.append(dCdBs)

        dCdWs = np.average(batch_dCdWs, axis=0) * self.learning_rate
        dCdBs = np.average(batch_dCdBs, axis=0) * self.learning_rate
        self.Ws = [W - dW for W, dW in zip(self.Ws, dCdWs)]
        self.Bs = [B - dB for B, dB in zip(self.Bs, dCdBs)]

    def forward_pass(self):
        self.Zs = []
        for i, (W, B) in enumerate(zip(self.Ws, self.Bs)):
            self.Zs.append(np.dot(W, self.Xs[i]) + B)
            self.Xs.append(sigmoid(self.Zs[i]))

    def backward_pass(self):
        dCdZ = ((self.Xs[-1] - self.Y) * sigmoid_prime(self.Zs[-1])).T
        dCdWs = [dCdZ * self.Xs[-2] + self.Ws[-1].T * self.regularization]
        dCdBs = [dCdZ]

        for i in range(len(self.sizes) - 2, 0, -1):
            dZdZ = self.Ws[i] * sigmoid_prime(self.Zs[i - 1]).T #dZdX * dXdZ
            dCdZ = np.dot(dCdZ, dZdZ)
            dCdW = dCdZ * self.Xs[i - 1] + self.Ws[i - 1].T * self.regularization #dCdZ * dZdW
            dCdWs.append(dCdW)
            dCdBs.append(dCdZ)

        dCdWs = [dCdW.T for dCdW in dCdWs[::-1]]
        dCdBs = [dCdB.T for dCdB in dCdBs[::-1]]
        return dCdWs, dCdBs

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_prime(Z):
    return (1 - sigmoid(Z)) * sigmoid(Z)

NN().run()
