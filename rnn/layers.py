import numpy as np
from utils import *
from basic_layers import *

def lstm_forward(xs, ys, ws, init_hc, task):
    (h, c), caches, loss_acc = init_hc(), {}, 0
    for t, (x, y) in enumerate(zip(xs, ys)):
        h, c, loss, _, caches[t] = lstm_full_forward(x, h, c, ws, y)
        loss_acc += loss
    return loss_acc, caches

def lstm_backward(caches, init_hc, init_ws):
    (dh, dc), dws = init_hc(), init_ws()
    for cache in reversed(list(caches.values())):
        cache_lstm, cache_affine, cache_sotfmax = cache
        ds = softmax_backward(cache_sotfmax)
        da, dwy = affine_backward(ds, cache_affine)
        dwh, dh, dc = lstm_backward_step(da + dh, dc, cache_lstm)
        dws += [dwh, dwy]
    return dws

def lstm_sample(x, ws, init_hc, vocab_size, task):
    (h, c), out = init_hc(1), {}
    for t in range(200):
        p = lstm_full_forward(x, h, c, ws, y=None)[3][:, 0]
        out[t] = np.random.choice(range(vocab_size), p=p)
        x = expand(task.one_of_k(num=out[t]))
    return list(out.values())

def lstm_val(xs, ys, ws, init_hc, task):
    (h, c), val_acc = init_hc(1), 0
    for t, (x, y) in enumerate(zip(xs, ys)):
        x = expand(x)
        h, c, _, p, _ = lstm_full_forward(x, h, c, ws, y=None)
        val_acc += p[np.argmax(y), 0] > .5
    return val_acc/1000

def lstm_full_forward(x, h, c, ws, y):
    wh, wy = ws
    h, c, cache_lstm = lstm_forward_step(x, h, c, wh)
    s, cache_affine = affine_forward(h, wy)
    loss, p, cache_softmax = softmax_forward(s, y)
    return h, c, loss, p, (cache_lstm, cache_affine, cache_softmax)
