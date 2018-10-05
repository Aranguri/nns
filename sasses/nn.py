from pprint import pprint
import numpy as np
import sys
sys.path.append('../')
import itertools
from basic_layers import *
from utils import *
from optimizer import *
from task3 import Task

batch_size = 100
embed_size = 3
hidden_size = 4
lr = 0.01
friction = .9
epoch_size = 10
seq_length = 5

np.random.seed(1)
task = Task(seq_length)
task.vocab_size = 4
wie = np.random.randn(embed_size, task.vocab_size)
weh = np.random.randn(hidden_size, embed_size) * np.sqrt(2 / (embed_size + 1))
whh = np.random.randn(hidden_size, hidden_size + 1) * np.sqrt(2 / (hidden_size + 1))
who = np.random.randn(task.vocab_size, hidden_size + 1) * np.sqrt(1 / (hidden_size + 1))
#tr_loss, val_acc, wie, weh, who = restore('nn-87.1-519')
ws = np.array([wie, weh, whh, who], dtype=object)
init_h = lambda: np.zeros((hidden_size, 1))
init_ws = lambda: ws - ws
dws = init_ws()
optimizer = SGD(lr)#Momentum(init_ws, lr, friction)
tr_loss, val_acc = {}, {}

#xs, ts = task.next_batch()
h, dh_prev = init_h(), init_h()
tr_loss, caches = {}, []
print (caches)
xs = np.random.randn(seq_length, task.vocab_size, 1)
ts = np.random.randint(task.vocab_size, size=(seq_length, 1))

def f(wie):
    h = init_h()
    for j, (x, t) in enumerate(zip(xs, ts)):
        embed, cache_wie = affine_forward(x, wie, bias=False)
        eh, cache_weh = affine_forward(embed, weh, bias=False)
        hh, cache_whh = affine_forward(h, whh)
        h, cache_h = act_fn_forward(eh + hh, tanh)
        o, cache_who = affine_forward(h, who)
        p = softmax_forward(o)
        tr_loss[j], cache_ce = ce_forward(p, t)
        caches.append((cache_wie, cache_weh, cache_whh, cache_h, cache_who, cache_ce))
    return dict_sum(tr_loss), caches

loss, caches = f(wie)

for cache in reversed(caches):
    cache_wie, cache_weh, cache_whh, cache_h, cache_who, cache_ce = cache
    do = softmax_ce_backward(cache_ce)
    dh, dwho = affine_backward(do, cache_who)
    deh_hh = act_fn_backward(dh + dh_prev, cache_h, tanh_prime)
    dh_prev, dwhh = affine_backward(deh_hh, cache_whh)
    dembed, dweh = affine_backward(deh_hh, cache_weh)
    dx, dwie = affine_backward(dembed, cache_wie)
    dws += [dwie, dweh, dwhh, dwho]

dwie, dweh, dwhh, dwho = dws
grad = eval_numerical_gradient(f, wie)
print (dwie)
print (grad)
print (rel_difference(grad, dwie))


'''
for i in itertools.count():
    xs, ts = task.next_batch()
    h, dh_prev = init_h(), init_h()
    caches, tr_loss[i] = {}, {}

    for j, (x, t) in enumerate(zip(xs, ts)):
        embed, cache_wie = affine_forward(x, wie, bias=False)
        eh, cache_weh = affine_forward(embed, weh, bias=False)
        hh, cache_whh = affine_forward(h, whh)
        h, cache_h = act_fn_forward(eh + hh, tanh)
        o, cache_who = affine_forward(h, who)
        p = softmax_forward(o)
        tr_loss[i][j], cache_ce = ce_forward(p, t)
        caches[j] = cache_wie, cache_weh, cache_whh, cache_h, cache_who, cache_ce

    for cache in caches.values():
        cache_wie, cache_weh, cache_whh, cache_h, cache_who, cache_ce = cache
        do = softmax_ce_backward(cache_ce)
        dh, dwho = affine_backward(do, cache_who)
        deh_hh = act_fn_backward(dh + dh_prev, cache_h)
        dh_prev, dwhh = affine_backward(deh_hh, cache_whh)
        dembed, dweh = affine_backward(deh_hh, cache_weh)
        dx, dwie = affine_backward(dembed, cache_wie)

    ws = optimizer.update(ws, [dwie, dweh, dwhh, dwho])

    print (dict_mean(tr_loss[i]))
'''
