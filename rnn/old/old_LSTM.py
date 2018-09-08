import itertools
import numpy as np
from utils import *
from text_env import TextEnv
from plotter import plot

hidden_size = 100
seq_length = 15
batch_size = 1
running_times = 10000 
learning_rate = 1e-3
b1, b2 = .9, .999
eps = 1e-6

tenv = TextEnv(seq_length, batch_size)
tenv.validation_data()
stop
vocab_size = tenv.vocab_size
wxh = np.random.randn(4*hidden_size, vocab_size) * 1e-2
whh = np.random.randn(4*hidden_size, hidden_size + 1) * 1e-2
why = np.random.randn(vocab_size, hidden_size + 1) * 1e-2
init_hc = lambda: (add_bias_term(np.zeros((hidden_size, batch_size))), np.zeros((hidden_size, batch_size)))
init_ws = lambda: (np.zeros_like(wxh), np.zeros_like(whh), np.zeros_like(why))
ws, mws, vws = [wxh, whh, why], init_ws(), init_ws()
loss_history = {}

def forward(xs, ys):
    h, c = init_hc()
    cache = {}
    loss = 0
    for t in range(len(xs)):
        v = whh.dot(h) + wxh.dot(xs[t])
        pi, pf, po, pg = np.split(v, 4)
        i, f, o, g = sigmoid(pi), sigmoid(pf), sigmoid(po), np.tanh(pg)
        c = c * f + g * i
        tanh_c = np.tanh(c)
        h = add_bias_term(tanh_c * o)
        s = why.dot(h)
        exp_scores = np.exp(s)
        scores = exp_scores / exp_scores.sum(0)
        loss -= np.sum(np.log(scores[ys[t], range(len(ys[t]))]))
        cache[t] = pi, pf, po, pg, i, f, o, g, h, c, tanh_c, s, exp_scores, scores

    return loss, cache

def backward(xs, ys, cache):
    dwxh, dwhh, dwhy = init_ws()
    dh_prev, dc_prev = init_hc()

    for t in reversed(range(len(xs))):
        pi, pf, po, pg, i, f, o, g, h, c, tanh_c, s, exp_scores, scores = cache[t]
        next_h, next_c = cache[t-1][8:10] if t != 0 else init_hc()
        ds = scores
        ds[ys[t], range(len(ys[t]))] -= 1
        dwhy += ds.dot(h.T)
        dh = why.T.dot(ds) + dh_prev
        dh = dh[:-1] #Remove bias term

        do = tanh_c * dh
        dtanh_c = o * dh
        dc = tanh_prime(c) * dtanh_c + dc_prev
        dc_prev = f * dc
        df = next_c * dc
        dg = i * dc
        di = g * dc

        dpi = sigmoid_prime(pi) * di
        dpf = sigmoid_prime(pf) * df
        dpo = sigmoid_prime(po) * do
        dpg = tanh_prime(pg) * dg

        dv = np.concatenate((dpi, dpf, dpo, dpg))

        dwhh += dv.dot(next_h.T)
        dwxh += dv.dot(xs[t].T)
        dh_prev = whh.T.dot(dv)

    return dwxh, dwhh, dwhy

def run(learning_rate):
    for i in range(1, running_times):
        xs, ys = tenv.next_batch()
        loss_history[i], cache = forward(xs, ys)
        dws = backward(xs, ys, cache)

        for w, dw, mw, vw in zip(ws, dws, mws, vws):
            #mw = b1 * mw + (1 - b1) * dw
            #mw /= (1 - b1 ** i)
            #vw = b2 * vw + (1 - b2) * dw ** 2
            #vw /= (1 - b2 ** i)
            w -= learning_rate * dw#mw / (np.sqrt(vw) + eps)

        if i % 100 == 0:
            x, (h, c) = tenv.rand_x(), init_hc()
            h, c = expand(h[:, 0]), expand(c[:, 0])
            text = ''

            for t in range(400):
                v = whh.dot(h) + wxh.dot(x)
                pi, pf, po, pg = np.split(v, 4)
                c = c * sigmoid(pf) + np.tanh(pg) * sigmoid(pi)
                h = add_bias_term(np.tanh(c) * sigmoid(po))
                s = why.dot(h)
                scores = np.exp(s) / np.exp(s).sum()
                x = np.random.choice(range(vocab_size), p=scores[:, 0])
                text += tenv.i_to_char[x]
                x = expand(tenv.one_hot(num=x))
            print ('\n-------\nLoss: {}\n{}\n------\n'.format(loss_history[i], text))
    return loss_history
