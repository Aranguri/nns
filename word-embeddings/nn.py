import numpy as np
import sys
sys.path.append('/home/aranguri/Desktop/dev/nns/rnn')
from basic_layers import *
from utils import *
from optimizer import *
from task import Task

batch_size = 100
num_words = 3
embed_size = 100
hidden_size = 30
friction = .9
lr = 1e-2

task = Task(batch_size)
wie = np.random.randn(embed_size, task.vocab_size) * 1e-3
weh = np.random.randn(hidden_size, embed_size * num_words + 1) * 1e-3
who = np.random.randn(task.vocab_size, hidden_size + 1) * 1e-3
ws = np.array([wie, weh, who], dtype=object)
init_ws = lambda: ws - ws
optimizer = Momentum(init_ws, friction, lr)
tr_loss, val_acc = {}, {}

for i in itertools.count():
    xs, ts = task.next_batch()
    embed, cache_embed = embed_forward(xs, wie)
    a, cache_act_fn = act_fn_forward(embed, weh, relu)
    s, cache_act_fn_2 = act_fn_forward(a, who, identity)
    tr_loss[i], p, cache_sm = softmax_forward(s, ts)

    ds = softmax_backward(cache_sm)
    da, dwho = act_fn_backward(ds, relu_prime, cache_act_fn_2)
    dx, dweh = act_fn_backward(da, identity_prime, cache_act_fn)
    dwie, dembed = embed_backward(dx, cache_embed)

    ws = optimizer.update(ws, [dwie, dweh, who])

    val_xs, val_ts = task.val_data()
    embed, _ = embed_forward(val_xs, wie)
    a, _ = act_fn_forward(embed, weh, relu)
    s, _ = act_fn_forward(a, who, identity)
    p = softmax_forward(s)
    val_acc[i] = (val_ts == p.argmax(0)).mean()

    print(val_acc[i])
    plot(val_acc)

'''
Some thigns take time, went out .172 at iteration 1000.
.181 (tanh, no hidden layers, SGD, 1e-3 wi, 1e-2 lr.)
best acc shofar: .186
comprar noise cancelling headphones
try with and without weight-tight
is a bias necessary in the embeddings?
see what's inside dembed
'''
