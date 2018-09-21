import numpy as np
from utils import *
from basic_layers import *

def lstm_forward(xs, ys, ws, init_hscs):
    (hs, cs), caches, losses = init_hscs(), {}, {}
    for t, (x, y) in enumerate(zip(xs, ys)):
        hs, cs, losses[t], _, caches[t] = lstm_full_forward(x, hs, cs, ws, y)
    return dict_sum(losses), caches

def lstm_backward(caches, init_hscs, init_whs):
    (dhs, dcs), dwhs_acc, dwy = init_hscs(), init_whs(), {}
    for t, (cache_lstms, cache_affine, cache_softmax) in enumerate(reversed(list(caches.values()))):
        ds = softmax_backward(cache_softmax)
        above_dh, dwy[t] = affine_backward(ds, cache_affine)
        for i, cache_lstm in reversed(list(enumerate(cache_lstms))):
            #dhs, dcs in assign => t. dhs, dcs in params => t + 1.
            dwhs, above_dh, dhs[i], dcs[i] = lstm_backward_step(above_dh + dhs[i], dcs[i], cache_lstm)
            dwhs_acc[i] += dwhs
    return dwhs_acc, dict_sum(dwy)

def lstm_sample(x, ws, init_hscs, vocab_size, task):
    (hs, cs), out = init_hscs(1), {}
    for t in range(200):
        p = lstm_full_forward(x, hs, cs, ws, y=None)[3][:, 0]
        out[t] = np.random.choice(range(vocab_size), p=p)
        x = expand(task.one_of_k(num=out[t]))
    return list(out.values())

def lstm_val(xs, ys, ws, init_hscs, task):
    (hs, cs), val_acc = init_hscs(1), 0
    for t, (x, y) in enumerate(zip(xs, ys)):
        x = expand(x)
        hs, cs, _, p, _ = lstm_full_forward(x, hs, cs, ws, y=None)
        val_acc += p[np.argmax(y), 0] > .5
    return val_acc/1000

def lstm_full_forward(x, hs_prev, cs_prev, ws, y):
    whs, wy = ws
    hs, cs, cache_lstms = {-1: x}, {}, {}
    for i, (h_prev, c_prev, wh) in enumerate(zip(hs_prev, cs_prev, whs)):
        hs[i], cs[i], cache_lstms[i] = lstm_forward_step(hs[i-1], h_prev, c_prev, wh)
    s, cache_affine = affine_forward(hs[len(whs) - 1], wy)
    loss, p, cache_softmax = softmax_forward(s, y)
    hs = list(hs.values())[1:]
    cs = list(cs.values())
    cache_lstms = list(cache_lstms.values())
    return hs, cs, loss, p, (cache_lstms, cache_affine, cache_softmax)
