import numpy as np
from utils import *

def step_forward_lstm(x, h, c, wxh, whh):
    v = wxh.dot(x) + whh.dot(add_bias(h))
    pi, pf, po, pg = np.split(v, 4)
    i, f, o, g = sigmoid(pi), sigmoid(pf), sigmoid(po), np.tanh(pg)
    c = c * f + g * i
    tanh_c = np.tanh(c)
    h = tanh_c * o
    cache = x, whh, pi, pf, po, pg, i, f, o, g, c, tanh_c
    return h, c, cache

def step_backward_lstm(dout, cache, dc_prev, h_prev, c_prev):
    x, whh, pi, pf, po, pg, i, f, o, g, c, tanh_c = cache

    do = tanh_c * dout
    dtanh_c = o * dout
    dc = tanh_prime(c) * dtanh_c + dc_prev
    dc_prev = f * dc
    df = c_prev * dc
    dg = i * dc
    di = g * dc

    dpi = sigmoid_prime(pi) * di
    dpf = sigmoid_prime(pf) * df
    dpo = sigmoid_prime(po) * do
    dpg = tanh_prime(pg) * dg

    dv = np.concatenate((dpi, dpf, dpo, dpg))

    dwxh = dv.dot(x.T)
    dwhh = dv.dot(add_bias(h_prev).T)
    dh_prev = whh.T.dot(dv)#[:-1]

    return dwxh, dwhh, dh_prev, dc_prev

def step_forward_affine(x, w):
    x = add_bias(x)
    s = w.dot(x)
    return s, (x, w)

def step_backward_affine(dout, cache):
    x, w = cache
    dx = w.T.dot(dout)[:-1] # :-1 removes the bias
    dw = dout.dot(x.T)
    return dx, dw

def softmax(s, y):
    exp_scores = np.exp(s)
    scores = exp_scores / exp_scores.sum(0)
    loss = -np.sum(np.log(scores[y, range(len(y))]))
    ds = scores
    ds[y, range(len(y))] -= 1
    return ds, loss

def forward_lstm(xs, ys, wxh, whh, why, init_hc):
    h, c = init_hc()
    s, cache, loss = {}, {}, {}
    for t in range(len(xs)):
        prev_h, prev_c = h, c
        h, c, lstm_cache = step_forward_lstm(xs[t], h, c, wxh, whh)
        s[t], affine_cache = step_forward_affine(h, why)
        ds, loss[t] = softmax(s[t], ys[t])
        cache[t] = prev_h, prev_c, h, c, lstm_cache, s[t], affine_cache, ds
    return np.sum([v for v in loss.values()]), cache

def backward_lstm(ys, cache, init_hc, init_ws):
    dh_prev, dc_prev = init_hc()
    grads = init_ws()
    loss = {}
    for t in reversed(range(len(ys))):
        prev_h, prev_c, h, c, lstm_cache, s, affine_cache, ds = cache[t]
        h_prev, c_prev = cache[t - 1][:2] if t != 0 else init_hc()
        dh, dwhy = step_backward_affine(ds, affine_cache)
        dh += dh_prev
        dwxh, dwhh, dh_prev, dc_prev = step_backward_lstm(dh, lstm_cache, dc_prev, h_prev, c_prev)
        grads = [acc_dw + dw for acc_dw, dw in zip(grads, [dwxh, dwhh, dwhy])]
    loss = np.sum([v for v in loss.values()])
    return grads, loss

def lstm_sample(x, wxh, whh, why, hidden_size, vocab_size):
    h, c = np.zeros((hidden_size, 1)), np.zeros((hidden_size, 1))
    s = {}
    for t in range(100):
        h, c, _ = step_forward_lstm(x, h, c, wxh, whh)
        s[t], _ = step_forward_affine(h, why)
        s[t] = np.random.choice(range(vocab_size), p=normalize(s[t]))
        x = expand(one_hot(s[t], vocab_size))
    return s.values()

hidden_size = 2
batch_size = 1
x = np.ones(hidden_size, batch_size)
h = np.ones((hidden_size, batch_size))
c = np.zeros((hidden_size, batch_size))
wxh = np.ones((4 * hidden_size, hidden_size))
whh = np.ones((4 * hidden_size, hidden_size))

dout = np.ones((hidden_size, batch_size))
dc_prev = np.zeros((hidden_size, batch_size))

def flstm(whh):
    return step_forward_lstm(x, h, c, wxh, whh)

s = np.random.randn(hidden_size, batch_size)
y = np.random.randint(0, hidden_size, size=(batch_size,))

def fsoft(s):
    return softmax(s, y)

grad = eval_numerical_gradient(fsoft, s)
print ('Numerical: ', grad)
ds, loss = fsoft(s)
print ('Analytical', ds)
print (rel_difference(grad, ds))
