import numpy as np
import sys
sys.path.append('/home/aranguri/Desktop/dev/nns/rnn')
from basic_layers import *
from utils import *
from optimizer import *
from task import Task

batch_size = 100
num_words = 5
embed_size = 100
hidden_size = 10
friction = .9
lr = 1

task = Task(batch_size)
wie = np.random.randn(embed_size, task.vocab_size) * 1e-1
weh = np.random.randn(hidden_size, embed_size * num_words + 1) * 1e-1
who = np.random.randn(1, hidden_size + 1) * 1e-1
ws = np.array([wie, weh, who], dtype=object)
init_ws = lambda: ws - ws
optimizer = Momentum(init_ws, friction, lr)
tr_loss, val_acc = {}, {}

for i in itertools.count():
    xs, ts = task.next_batch()
    embed, cache_embed = embed_forward(xs, wie)
    a, cache_act_fn = act_fn_forward(embed, weh, relu)
    s, cache_act_fn_2 = act_fn_forward(a, who, sigmoid)
    tr_loss[i], cache_ce = ce_forward(s, ts)

    ds = ce_backward(cache_ce)
    da, dwho = act_fn_backward(ds, sigmoid_prime, cache_act_fn_2)
    dx, dweh = act_fn_backward(da, relu_prime, cache_act_fn)
    dwie, dembed = embed_backward(dx, cache_embed)

    ws = optimizer.update(ws, [dwie, dweh, dwho])

    val_xs, val_ts = task.val_data()
    embed, _ = embed_forward(val_xs, wie)
    a, _ = act_fn_forward(embed, weh, relu)
    s, _ = act_fn_forward(a, who, sigmoid)
    val_acc[i] = ((s > .5) == val_ts).mean()

    print(val_acc[i])
    plot(val_acc)
