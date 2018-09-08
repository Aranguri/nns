import numpy as np
from utils import psh, one_hot, expand, tanh_prime

learning_rate = 3e-3
hidden_size = 150
seq_length = 25
epsilon = 1e-6

data = open('input.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

char_to_i = {char: i for i, char in enumerate(chars)}
i_to_char = {i: char for i, char in enumerate(chars)}

wxh = np.random.randn(vocab_size, hidden_size) * .01
whh = np.random.randn(hidden_size, hidden_size) * .01
why = np.random.randn(hidden_size, vocab_size) * .01
bh = np.zeros((1, hidden_size))
by = np.zeros((1, vocab_size))

def loss_fun(inputs, targets, h_prev):
    tanh_arg, softmax, h = {}, {}, {}
    dwxh, dwhh, dwhy = np.zeros_like(wxh), np.zeros_like(whh), np.zeros_like(why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    loss = 0
    h[-1] = h_prev

    for t in range(len(inputs)):
        tanh_arg[t] = inputs[t].dot(wxh) + h[t - 1].dot(whh) + bh
        h[t] = np.tanh(tanh_arg[t])
        y = h[t].dot(why) + by
        exp_y = np.exp(y)
        softmax[t] = exp_y / exp_y.sum()
        correct_scores = (softmax[t] * targets[t]).sum(1)
        loss += np.sum(-np.log(correct_scores))

    for t in reversed(range(len(inputs))):
        df = softmax[t] - targets[t]
        dby += df.sum(0, keepdims=True)
        dwhy += h[t].T.dot(df)

        dh = why.dot(df.T)
        dtanh_arg = tanh_prime(tanh_arg[t]) * dh.T
        dbh += dtanh_arg
        dwxh += expand(inputs[t]).dot(dtanh_arg)
        dwhh += h[t - 1].T.dot(dtanh_arg)

    return loss, dwxh, dwhh, dwhy, dbh, dby, h[-1]

def sample():
    xs = {0: one_hot(char_to_i[np.random.choice(chars)], vocab_size)}
    h = {-1: np.zeros((1, hidden_size))}
    print ('\n\n')
    for t in range(100):
        h[t] = np.tanh(xs[t].dot(wxh) + h[t - 1].dot(whh) + bh)
        y = h[t].dot(why) + by
        softmax = np.exp(y) / np.exp(y).sum()
        next_i = np.random.choice(vocab_size, p=softmax[0])
        xs[t + 1] = one_hot(next_i, vocab_size)
        print(i_to_char[next_i], end='')
    print ('\n\n')

i = 0
h_prev = np.zeros((1, hidden_size))
smooth_loss = 0

while True:
    if i > data_size:
        i = 0
        h_prev = np.zeros((1, hidden_size))

    inputs = data[i:i + seq_length]
    targets = data[i + 1:i + 1 + seq_length]
    inputs = np.array([one_hot(char_to_i[char], vocab_size) for char in inputs])
    targets = np.array([one_hot(char_to_i[char], vocab_size) for char in targets])
    loss, dwxh, dwhh, dwhy, dbh, dby, h_prev = loss_fun(inputs, targets, h_prev)

    #print (loss)
    if smooth_loss == 0: smooth_loss = loss
    smooth_loss = .999 * smooth_loss + .001 * loss
    for param, dparam in zip([wxh, whh, why, bh, by], [dwxh, dwhh, dwhy, dbh, dby]):
        param -= learning_rate * dparam

    if i % 100 == 0:
        sample()
        print (smooth_loss)

    i += 1


'''
Las perlitas:
--
e
Iacricnse roing ereands
p poierrs dring endands
I procrastinate doing errands
I prico errarostinas
---
'''
