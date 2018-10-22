import numpy as np
import sys
sys.path.append('../')
from basic_layers import *
from utils import *
from optimizer import *
from task2 import Task

batch_size = 100
embed_size = 50
hidden_size = 200
lr = 0.3
friction = .9
b1, b2 = .9, .999
epoch_size = 10

task = Task(batch_size)
wie = np.random.randn(embed_size, task.vocab_size) * np.sqrt(1/task.vocab_size)
weh = np.random.randn(hidden_size, embed_size * task.num_words + 1) * np.sqrt(2/(embed_size * task.num_words + 1))
who = np.random.randn(task.vocab_size, hidden_size + 1) * np.sqrt(1 / (hidden_size + 1))
#tr_loss, val_acc, wie, weh, who = restore('nn-87.1-519')
ws = np.array([wie, weh, who], dtype=object)
init_ws = lambda: ws - ws
optimizer = Momentum(init_ws, lr, friction)#SGD(lr)

for epoch in range(epoch_size):
    print (f'Epoch {epoch}')
    tr_loss, val_acc = {}, {}
    xs, ts = task.train_xs, task.train_ts

    for i, (x, t) in enumerate(zip(xs, ts)):
        x = task.one_of_k(x)
        embed, cache_embed = embed_forward(x, wie)
        s, cache_act_fn = act_fn_forward(embed, weh, sigmoid)
        u, cache_act_fn_2 = act_fn_forward(s, who, identity)
        p = softmax_forward(u)
        tr_loss[i], cache_ce = ce_forward(p, t)

        du = softmax_ce_backward(cache_ce)
        ds, dwho = act_fn_backward(du, identity_prime, cache_act_fn_2)
        da, dweh = act_fn_backward(ds, sigmoid_prime, cache_act_fn)
        dwie, dembed = embed_backward(da, cache_embed)
        ws = optimizer.update(ws, [dwie, dweh, dwho])

        if i % 100 == 0:
            print (f'Batch {i}. {dict_mean(tr_loss, -100)}')
            plot(tr_loss)

        if i % 1000 == 0:
            val_x, val_t = task.val_data()
            embed, _ = embed_forward(val_x, wie)
            s, _ = act_fn_forward(embed, weh, sigmoid)
            u, _ = act_fn_forward(s, who, identity)
            p = softmax_forward(u)
            loss, _ = ce_forward(p, val_t)
            print (f'Val err {loss}')
            print (f'Val acc: {np.mean(np.argmax(p, 0) == val_t)}')
            save(f'lt-{i}', tr_loss, val_acc, *ws)
