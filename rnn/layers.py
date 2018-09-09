import numpy as np
from utils import *

def lstm_forward(xs, ys, wxh, whh, why, init_hc):
    (h, c), caches, loss_acc = init_hc(), {}, 0
    for t, (x, y) in enumerate(zip(xs, ys)):
        h, c, loss, _, caches[t] = lstm_full_forward(x, h, c, wxh, whh, why, y)
        loss_acc += loss
    return loss_acc, caches

def lstm_backward(caches, init_hc, init_ws):
    (dh, dc), dws = init_hc(), init_ws()
    for cache in reversed(list(caches.values())):
        cache_lstm, cache_affine, cache_sotfmax = cache
        ds = softmax_backward(cache_sotfmax)
        da, dwhy = affine_backward(ds, cache_affine)
        dwxh, dwhh, dh, dc = lstm_backward_step(da + dh, dc, cache_lstm)
        dws += [dwxh, dwhh, dwhy]
    return dws

def lstm_sample(x, wxh, whh, why, init_hc, vocab_size, task):
    (h, c), out = init_hc(1), {}
    for t in range(200):
        p = lstm_full_forward(x, h, c, wxh, whh, why, y=None)[3]
        out[t] = np.random.choice(range(vocab_size), p=normalize(p))
        x = expand(task.one_hot(num=out[t]))
    return list(out.values())

def lstm_val(xs, ys, wxh, whh, why, init_hc, k, task):
    (h, c), val_acc = init_hc(1), 0
    for t, (x, y) in enumerate(zip(xs, ys)):
        x = expand(x)
        h, c, _, p, _ = lstm_full_forward(x, h, c, wxh, whh, why, y=None)
        top_k = np.argpartition(p[:, 0], -k)[-k:]
        val_acc += np.argmax(y) in top_k
    return val_acc

def lstm_full_forward(x, h, c, wxh, whh, why, y):
    h, c, cache_lstm = lstm_forward_step(x, h, c, wxh, whh)
    s, cache_affine = affine_forward(h, why)
    loss, p, cache_softmax = softmax_forward(s, y)
    return h, c, loss, p, (cache_lstm, cache_affine, cache_softmax)

def lstm_forward_step(x, h_prev, c_prev, wxh, whh):
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

def lstm_backward_step(dout, dc_out, cache):
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

def affine_forward(x, w):
    x = add_bias(x)
    s = w.dot(x)
    return s, (x, w)

def affine_backward(dout, cache):
    x, w = cache
    dx = w.T.dot(dout)[:-1] # :-1 removes the bias
    dw = dout.dot(x.T)
    return dx, dw

def softmax_forward(s, y=None):
    exp_s = np.exp(s)
    p = exp_s / exp_s.sum(0)
    if y is not None:
        correct_s = p[y, range(len(y))]
        loss = -np.sum(np.log(correct_s))
        cache = p, y
        return loss, p, cache
    else:
        return None, p, None

def softmax_backward(cache):
    p, y = cache
    p[y, range(len(y))] -= 1
    return p
