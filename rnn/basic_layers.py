import numpy as np
from utils import *

def lstm_forward_step(h_below, h_prev, c_prev, wh):
    #affine
    v = wh.dot(np.concatenate((h_below, add_bias(h_prev))))

    #gates
    pi, pf, po, pg = np.split(v, 4)
    i, f, o, g = sigmoid(pi), sigmoid(pf), sigmoid(po), np.tanh(pg)

    #c
    c = f * c_prev + i * g

    #h
    tanh_c = np.tanh(c)
    h = tanh_c * o

    cache = h_below, h_prev, wh, pi, pf, po, pg, i, f, o, g, c_prev, c, tanh_c
    return h, c, cache

def lstm_backward_step(dout, dc_out, cache):
    h_below, h_prev, wh, pi, pf, po, pg, i, f, o, g, c_prev, c, tanh_c = cache

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
    dwh_below = dv.dot(h_below.T)
    dwh_prev = dv.dot(add_bias(h_prev).T)
    dwh = np.concatenate((dwh_below, dwh_prev), 1)
    #ps(wh)

    wh_below = wh[:, :len(h_below)] #first h items of wh
    #print (len(h_below))
    #ps(wh_below)
    wh_prev = wh[:, len(h_below):] #last h items of wh
    dh_below = wh_below.T.dot(dv)
    dh_prev = remove_bias(wh_prev.T).dot(dv)
    return dwh, dh_below, dh_prev, dc_prev

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
