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
embed_size = 50
hidden_size = 150
lr = 0.01
friction = .9
epoch_size = 10
seq_length = 10

task = Task(seq_length, batch_size)
wie = np.random.randn(embed_size, task.vocab_size)
weh = np.random.randn(hidden_size, embed_size) * np.sqrt(2 / (embed_size + 1))
whh = np.random.randn(hidden_size, hidden_size + 1) * np.sqrt(2 / (hidden_size + 1))
who = np.random.randn(task.vocab_size, hidden_size + 1) * np.sqrt(1 / (hidden_size + 1))
ws = np.array([wie, weh, whh, who], dtype=object)
init_h = lambda: np.zeros((hidden_size, batch_size))
init_ws = lambda: ws - ws

optimizer = Momentum(init_ws, lr, friction)
tr_loss, tr_acc, val_acc = {}, {}, {}
#tr_loss, val_acc, wie, weh, who = restore('nn-87.1-519')

h = init_h()
for i in itertools.count():
    xs, ts = task.next_batch()
    dh_prev, dws = init_h(), init_ws()
    tr_loss[i], tr_acc[i], caches = {}, {}, []

    for j, (x, t) in enumerate(zip(xs, ts)):
        embed, cache_wie = affine_forward(x, wie, bias=False)
        eh, cache_weh = affine_forward(embed, weh, bias=False)
        hh, cache_whh = affine_forward(h, whh)
        h, cache_h = act_fn_forward(eh + hh, tanh)
        o, cache_who = affine_forward(h, who)
        p = softmax_forward(o)
        tr_loss[i][j], cache_ce = ce_forward(p, t)
        tr_acc[i][j] = accuracy_forward(p, t)
        # print (np.argmax(x), t[0], np.argmax(p))
        caches.append((cache_wie, cache_weh, cache_whh, cache_h, cache_who, cache_ce))

    for cache in reversed(caches):
        cache_wie, cache_weh, cache_whh, cache_h, cache_who, cache_ce = cache
        do = softmax_ce_backward(cache_ce)
        dh, dwho = affine_backward(do, cache_who)
        deh_hh = act_fn_backward(dh + dh_prev, cache_h, tanh_prime)
        dh_prev, dwhh = affine_backward(deh_hh, cache_whh)
        dembed, dweh = affine_backward(deh_hh, cache_weh)
        dx, dwie = affine_backward(dembed, cache_wie)
        dws += [dwie, dweh, dwhh, dwho]

    ws = optimizer.update(ws, dws)

    if i % 10 == 0:
        x = xs[-1, :, 0].reshape(-1, 1)
        h_sample = init_h()[:, 0].reshape(-1, 1)
        for j in range(100):
            embed, _ = affine_forward(x, wie, bias=False)
            eh, _ = affine_forward(embed, weh, bias=False)
            hh, _ = affine_forward(h_sample, whh)
            h_sample, _ = act_fn_forward(eh + hh, tanh)
            o, _ = affine_forward(h_sample, who)
            p = softmax_forward(o)
            i = weighted_random(p)
            print (task.ixs_to_words(i), end=' ')
        print ('\n')

    accs = [dict_mean(l) for l in tr_acc.values()]
    losses = [dict_mean(l) for l in tr_loss.values()]
    plot(accs)
    print(f'Accs {np.mean(accs)}')
