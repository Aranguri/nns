import numpy as np
import itertools

data = open('input.txt', 'r').read()
chars = list(set(data))
data_size, x_size = len(data), len(chars)
char_to_i = {c: i for c, i in enumerate(chars)}
i_to_char = {i: c for c, i in enumerate(chars)}

h_size = 100
seq_length = 25
learning_rate = 1e-1

W_xh = np.random.randn(h_size, x_size) * .01
W_hh = np.random.randn(h_size, h_size) * .01
W_hy = np.random.randn(x_size, h_size) * .01
B_h = np.zeros((self.h_size, 1))
B_y = np.zeros((self.x_size, 1))

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(W_xh), np.zeros_like(W_hh), np.zeros_like(W_hy)
mbh, mby = np.zeros_like(B_h), np.zeros_like(B_y)
smooth_loss = -np.log(1.0 / x_size) * seq_length

def main():
    for i in itertools.count():
        if p + seq_length + 1 >= data_size or n == 0:
            h_prev = np.zeros(h_size)
            p = 0

        inputs = [char_to_i[ch] for ch in data[p:p + seq_length]]
        targets = [char_to_i[ch] for ch in data[p + 1:p + seq_length + 1]]

        loss, dCdWxh, dCdWhh, dCdWhy, dCdBh, dCdBy = sgd(inputs, targets, h_prev)

        smooth_loss = smooth_loss * .999 + loss * .001

        if i % 100 == 0:
            sample_i = sample(h_prev, inputs[0], 200)
            print ('Iteration: {}. Loss: {}'.format(i, smooth_loss))
            print (''.join([i_to_char[j] for j in sample_i]), end='\n\n')

def sgd(inputs, targets, h_prev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    #forward
    for t in range(len(inputs)):
        xs[t] = np.zeros((x_size, 1))
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(W_xh, xs[t]) + np.dot(W_hh, hs[t-1]) + B_h)
        ys[t] = np.dot(W_hy, hs[t]) + B_y
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) #need explanation
        loss += -np.log(ps[t][targets[t], 0]) #need explanation

    #backward
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):



def sample(h_prev, seed_i, times):
