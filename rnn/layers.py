import numpy as np
from utils import *

hidden_size = 3
batch_size = 2
seq_length = 4
x = np.random.randn(hidden_size, batch_size)
y = np.random.randint(hidden_size, size=(batch_size,))
wxh = np.random.randn(4 * hidden_size, hidden_size)
whh = np.random.randn(4 * hidden_size, hidden_size + 1)
why = np.random.randn(hidden_size, hidden_size + 1)
dout = np.random.randn(seq_length, hidden_size, batch_size)
init_hc = lambda: [np.zeros((hidden_size, batch_size))] * 2

def two_steps_forward(whh):
    (h, c), cache, loss_acc = init_hc(), {}, 0
    for t in range(seq_length):
        h, c, cache_lstm = new_lstm_forward(x, h, c, wxh, whh)
        s, cache_affine = step_forward_affine(h, why)
        loss, cache_sotfmax = forward_softmax(s, y)
        loss_acc += loss
        cache[t] = cache_lstm, cache_affine, cache_sotfmax
    return loss_acc, cache

def two_steps_backward(dout, cache):
    dh, dc = init_hc()
    sum_dwxh, sum_dwhh, sum_dwhy = np.zeros_like(wxh), np.zeros_like(whh), np.zeros_like(why)
    sum_dwhy = np.zeros_like(why)
    for t in reversed(range(seq_length)):
        cache_lstm, cache_affine, cache_sotfmax = cache[t]
        ds = backward_softmax(cache_sotfmax)
        da, dwhy = step_backward_affine(ds, cache_affine)
        dwxh, dwhh, dh, dc = new_lstm_backward(da + dh, dc, cache_lstm)
        sum_dwxh += dwxh
        sum_dwhh += dwhh
        sum_dwhy += dwhy
    return sum_dwxh, sum_dwhh, sum_dwhy

def new_lstm_forward(x, h_prev, c_prev, wxh, whh):
    #affine
    v = wxh.dot(x) + whh.dot(add_bias(h_prev))

    #gates
    pi, pf, po, pg = np.split(v, 4)
    i, f, o, g = sigmoid(pi), sigmoid(pf), sigmoid(po), np.tanh(pg)

    #c
    c = f * c_prev + i * g

    #h
    tanh_c = np.tanh(c)
    h = tanh_c * o

    cache = x, h_prev, wxh, whh, pi, pf, po, pg, i, f, o, g, c_prev, c, tanh_c
    return h, c, cache

def new_lstm_backward(dout, dc_out, cache):
    x, h_prev, wxh, whh, pi, pf, po, pg, i, f, o, g, c_prev, c, tanh_c = cache

    #h
    dtanh_c = o * dout
    do = tanh_c * dout
    dc = tanh_prime(c) * dtanh_c + dc_out

    #c
    df = c_prev * dc
    dc_prev = f * dc
    di = g * dc
    dg = i * dc

    #gates
    dpi = sigmoid_prime(pi) * di
    dpf = sigmoid_prime(pf) * df
    dpo = sigmoid_prime(po) * do
    dpg = tanh_prime(pg) * dg
    dv = np.concatenate((dpi, dpf, dpo, dpg))

    #affine
    dwxh = dv.dot(x.T)
    dwhh = dv.dot(add_bias(h_prev).T)
    dh_prev = remove_bias(whh.T).dot(dv)
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

def forward_softmax(s, y):
    exp_s = np.exp(s)
    p = exp_s / exp_s.sum(0)
    correct_s = p[y, range(len(y))]
    loss = -np.sum(np.log(correct_s))
    cache = p, y
    return loss, cache

def backward_softmax(cache):
    p, y = cache
    p[y, range(len(y))] -= 1
    return p

grad = eval_numerical_gradient(two_steps_forward, whh)
print ('Numerical: ', grad)
h, cache = two_steps_forward(whh)
dwxh, dwhh, dwhy = two_steps_backward(dout, cache)
print ('Analytical', dwhh)
print (rel_difference(grad, dwhh))












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
