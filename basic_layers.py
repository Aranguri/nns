import numpy as np
from utils import *

def affine_forward(x, w, bias=True):
    x = add_bias(x) if bias else x
    s = w.dot(x)
    return s, (x, w, bias)

def affine_backward(dout, cache):
    x, w, bias = cache
    dx = w.T.dot(dout)
    dx = dx[:-1] if bias else dx
    dw = dout.dot(x.T)
    return dx, dw

def act_fn_forward(s, fn):
    a = fn(s)
    return a, s

def act_fn_backward(dout, cache, fn_prime):
    s = cache
    ds = fn_prime(s) * dout
    return ds

def affine_fn_forward(x, w, fn):
    s, cache_affine = affine_forward(x, w)
    a, cache_act_fn = act_fn_forward(s, fn)
    return a, (cache_act_fn, cache_affine)

def affine_fn_backward(dout, cache, fn_prime):
    cache_act_fn, cache_affine = cache
    ds = act_fn_backward(dout, cache_act_fn, fn_prime)
    dx, dw = affine_backward(ds, cache_affine)
    return dx, dw

def embed_forward(x, w):
    embed = np.tensordot(x, w, (1, 1))
    embed = embed.swapaxes(1, 2)
    return embed, (x, w)

def embed_backward(dout, cache):
    x, w = cache
    dx = np.tensordot(dout, w, axes=((1), (0)))
    dx = dx.swapaxes(1, 2)
    dw = np.tensordot(dout, x, axes=((0, 2), (0, 2)))
    return dx, dw

def accuracy_forward(p, t):
    return np.mean(np.argmax(p, 0) == t)

def weighted_random(p):
    p = p[:, 0]
    return np.random.choice(np.arange(len(p)), p=p)

def softmax_forward(u):
    u_normal = u - np.amax(u)
    exp_u = np.exp(u_normal)
    p = exp_u / exp_u.sum(0)
    return p

def ce_forward(p, t):
    if p.shape[0] == 1:
        loss = -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))
    else:
        correct_s = p[t, range(len(t))]
        loss = np.mean(np.log(1 / (correct_s + 1e-30)))
    return loss, (p, t)

def softmax_ce_backward(cache):
    p, t = cache
    p[t, range(len(t))] -= 1
    p /= p.shape[1]
    return p

def ce_backward(cache):
    ps, ts = cache
    return (1 / ps.shape[1]) * ((ps - ts) / (ps * (1 - ps)))

def squared_cost_forward(x, t):
    loss = np.square(x - t).sum() / 2
    return loss, (x, t)

def squared_cost_backward(cache):
    x, t = cache
    return x - t

def regularization_forward(ws, reg):
    loss = (1/2) * reg * rec_sum([np.square(w) for w in ws])
    return loss, (reg, ws)

def regularization_backward(cache):
    reg, ws = cache
    return reg * np.array(ws)
