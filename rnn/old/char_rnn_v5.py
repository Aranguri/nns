import numpy as np
import itertools
import time
import os
from utils import psh, expand, tanh_prime

class RNN:
    restore_enabled = False

    def __init__(self, lr=1e-3, hidden_size=100, running_times=1000, seq_length=25, k=3, update='sgd', mode='train-acc', exp_name='untitled'):
        self.learning_rate = lr
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.running_times = running_times
        self.k = k #Number of tries that the nn has for calculating the accuracy
        self.update = update
        self.mode = mode
        self.exp_name = exp_name
        self.epsilon = 1e-8
        self.beta1 = .9
        self.beta2 = .999
        self.reg = 0#5e-4

        self.data = open('inputs/input.txt', 'r').read()
        self.train_data, self.val_data = self.data[:900000], self.data[900000:1000000]
        self.train_size, self.val_size = len(self.train_data),len(self.val_data)
        self.chars = sorted(list(set(self.data)))
        self.vocab_size = len(self.chars)
        self.char_to_i = {char: i for i, char in enumerate(self.chars)}
        self.i_to_char = np.array(self.chars)

        self.wxh = np.random.randn(self.vocab_size, self.hidden_size) * .01
        self.whh = np.random.randn(self.hidden_size, self.hidden_size) * .01
        self.why = np.random.randn(self.hidden_size, self.vocab_size) * .01
        self.bh = np.zeros((1, self.hidden_size))
        self.by = np.zeros((1, self.vocab_size))

        self.train_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.val_accuracy = []

        if self.restore_enabled: self.restore()

    def run(self):
        mwxh, mwhh, mwhy = np.zeros_like(self.wxh), np.zeros_like(self.whh), np.zeros_like(self.why)
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by)
        vwxh, vwhh, vwhy = np.zeros_like(self.wxh), np.zeros_like(self.whh), np.zeros_like(self.why)
        vbh, vby = np.zeros_like(self.bh), np.zeros_like(self.by)

        for i in range(self.running_times):
            n = (i * self.seq_length) % (self.train_size - self.seq_length)
            if n < self.seq_length:
                #This means either we are starting or we went through all the dataset
                self.h_prev = np.zeros((1, self.hidden_size))

            inputs = self.train_data[n:n + self.seq_length]
            targets = self.train_data[n + 1:n + self.seq_length + 1]
            inputs = np.array([self.one_hot(char) for char in inputs])
            targets = np.array([self.one_hot(char) for char in targets])
            dwxh, dwhh, dwhy, dbh, dby = self.loss_fun(inputs, targets)

            for param, dparam, mparam, vparam in zip([self.wxh, self.whh, self.why, self.bh, self.by],
                                                     [dwxh, dwhh, dwhy, dbh, dby],
                                                     [mwxh, mwhh, mwhy, mbh, mby],
                                                     [vwxh, vwhh, vwhy, vbh, vby]):
                np.clip(dparam, -5, 5, out=dparam)
                if self.update == 'adam':
                    mparam = self.beta1 * mparam + (1 - self.beta1) * dparam
                    vparam = self.beta2 * vparam + (1 - self.beta2) * dparam ** 2
                    mparam = mparam / (1 - self.beta1 ** (i + 1))
                    vparam = vparam / (1 - self.beta2 ** (i + 1))
                    param -= self.learning_rate * mparam / (np.sqrt(vparam) + self.epsilon)
                else:
                    param -= self.learning_rate * dparam

            self.wxh -= self.reg * self.wxh
            self.whh -= self.reg * self.whh
            self.why -= self.reg * self.why

            if i % 100 == 0:
                self.show_state(i)

    def loss_fun(self, inputs, targets):
        tanh_arg, softmax, h = {}, {}, {-1: self.h_prev}
        dwxh, dwhh, dwhy = np.zeros_like(self.wxh), np.zeros_like(self.whh), np.zeros_like(self.why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        loss = 0

        for t in range(len(inputs)):
            tanh_arg[t] = inputs[t].dot(self.wxh) + h[t - 1].dot(self.whh) + self.bh
            h[t] = np.tanh(tanh_arg[t])
            y = h[t].dot(self.why) + self.by
            exp_y = np.exp(y)
            softmax[t] = exp_y / exp_y.sum()
            correct_scores = (softmax[t] * targets[t]).sum(1)

        for t in reversed(range(len(inputs))):
            df = softmax[t] - targets[t]
            dby += df
            dwhy += h[t].T.dot(df)

            dh = self.why.dot(df.T)
            dtanh_arg = tanh_prime(tanh_arg[t]) * dh.T
            dbh += dtanh_arg
            dwxh += expand(inputs[t]).dot(dtanh_arg)
            dwhh += h[t - 1].T.dot(dtanh_arg)

        self.h_prev = h[-1]
        return dwxh, dwhh, dwhy, dbh, dby

    def forward(self, x, h_prev, target=None):
        h = np.tanh(x.dot(self.wxh) + h_prev.dot(self.whh) + self.bh)
        y = h.dot(self.why) + self.by
        softmax = np.exp(y) / np.exp(y).sum()
        loss = -y[0, target] + np.log(np.sum(np.exp(y))) if target != None else None
        return h, y, softmax, loss

    def show_state(self, it):
        if 'train' in self.mode:
            h = np.zeros((1, self.hidden_size))
            i = np.random.randint(8000) * 100
            self.train_accuracy.append(0)
            total_loss = 0

            for char, next_char in zip(self.train_data[i:i+1000], self.train_data[i+1:i+1001]):
                h, y, _, loss = self.forward(self.one_hot(char), h, self.char_to_i[next_char])
                total_loss += loss + self.reg * .5 * (np.sum(self.wxh**2) + np.sum(self.whh**2) + np.sum(self.why**2))
                topk = np.argpartition(y[0], -self.k)[-self.k:]
                self.train_accuracy[-1] += next_char in self.i_to_char[topk]

            self.train_loss.append(total_loss / 1000)
            self.train_accuracy[-1] /= 1000
            #print ('Train loss: {:.2f}. Train acc: {:.2f}. It: {}'.format(self.train_loss[-1], self.train_accuracy[-1], it))

        if 'val' in self.mode:
            h = np.zeros((1, self.hidden_size))
            i = np.random.randint(900) * 100
            self.val_accuracy.append(0)
            total_loss = 0

            for char, next_char in zip(self.val_data[i:i+1000], self.val_data[i+1:i+1001]):
                h, y, _, loss = self.forward(self.one_hot(char), h, self.char_to_i[next_char])
                total_loss += loss + self.reg * .5 * (np.sum(self.wxh**2) + np.sum(self.whh**2) + np.sum(self.why**2))
                topk = np.argpartition(y[0], -self.k)[-self.k:]
                self.val_accuracy[-1] += next_char in self.i_to_char[topk]

            self.val_loss.append(total_loss / 1000)
            self.val_accuracy[-1] /= 1000
            print ('Val loss:   {:.2f}. Val acc:   {:.2f}. It: {}'.format(self.val_loss[-1], self.val_accuracy[-1], it))

        if 'sample' in self.mode:
            xs = {0: self.one_hot(np.random.choice(self.chars))}
            h = {-1: np.zeros((1, self.hidden_size))}
            for t in range(200):
                h[t], y, softmax, _ = self.forward(xs[t], h[t - 1])
                next_i = np.random.choice(self.vocab_size, p=softmax[0])
                xs[t + 1] = self.one_hot(self.i_to_char[next_i])
            print (''.join([self.i_to_char[x.argmax()] for x in xs.values()]))
            print ('\n\n')

        if 'save' in self.mode and it % 1000 == 900:
            self.save()


    def one_hot(self, char):
        array = np.zeros((self.vocab_size))
        array[self.char_to_i[char]] = 1
        return array

    def save(self):
        path = 'savings/{}'.format(self.exp_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, hidden_size=self.hidden_size, wxh=self.wxh, whh=self.whh, why=self.why,
                 bh=self.bh, by=self.by, train_loss=self.train_loss, val_loss=self.val_loss,
                 train_accuracy=self.train_accuracy, val_accuracy=self.val_accuracy)

    def restore(self):
        npzfile = np.load('savings/{}.npz'.format(self.exp_name))
        self.hidden_size, self.wxh, self.whh, self.why, self.bh, self.by = [item for _, item in npzfile.items()[:6]]
        self.train_loss, self.val_loss, self.train_accuracy, self.val_accuracy = [list(item) for _, item in npzfile.items()[6:]]
        print ('Net restored')
