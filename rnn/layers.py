import numpy as np
from utils import *
from basic_layers import *

def lstm_forward(xs, ys, whs, wy, init_hscs, task):
    (hs, cs), caches, loss_acc = init_hscs(), {}, 0
    for t, (x, y) in enumerate(zip(xs, ys)):
        hs, cs, loss, _, caches[t] = lstm_full_forward(x, hs, cs, whs, wy, y)
        loss_acc += loss
    return loss_acc, caches

def lstm_backward(caches, init_hscs, init_whs, init_wy):
    (dhs, dcs) = init_hscs()
    dwhs_acc, dwy_acc = init_whs(), init_wy()
    for cache in reversed(list(caches.values())):
        cache_lstms, cache_affine, cache_sotfmax = cache
        ds = softmax_backward(cache_sotfmax)
        da, dwy = affine_backward(ds, cache_affine)
        new_dhs, new_dcs = init_hscs()
        for i, cache_lstm in reversed(list(enumerate(cache_lstms))):
            above_dh = new_dhs[i+1] if i != len(cache_lstms) - 1 else da #Same timestep, one layer above
            next_dh, next_dc = dhs[i], dcs[i]#Same layer, next timestep
            dwhs, new_dhs[i], new_dcs[i] = lstm_backward_step(above_dh + next_dh, next_dc, cache_lstm)
            dwhs_acc[i] += dwhs
        dhs, dcs = new_dhs, new_dcs
        dwy_acc += dwy
    return dwhs_acc, dwy_acc

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

def lstm_full_forward(x, hs_prev, cs_prev, whs, wy, y):
    hs, cs, cache_lstms = {-1: x}, {}, {}
    for i, (h_prev, c_prev, wh) in enumerate(zip(hs_prev, cs_prev, whs)):
        hs[i], cs[i], cache_lstms[i] = lstm_forward_step(hs[i-1], h_prev, c_prev, wh)
    s, cache_affine = affine_forward(hs[len(whs) - 1], wy)
    loss, p, cache_softmax = softmax_forward(s, y)
    hs = list(hs.values())[1:]
    cs = list(cs.values())
    cache_lstms = list(cache_lstms.values())
    return hs, cs, loss, p, (cache_lstms, cache_affine, cache_softmax)
