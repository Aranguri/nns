import numpy as np
import sys
sys.path.append('/home/aranguri/Desktop/dev/nns/rnn')
from basic_layers import *
from utils import *
from optimizer import *
from task import Task

batch_size = 4
num_words = 5
embed_size = 1
hidden_size = 1
friction = .9
lr = 1e-3

task = Task(batch_size)
wie = np.random.randn(embed_size, task.vocab_size) * 1e-3
weh = np.random.randn(hidden_size, embed_size * num_words + 1) * 1e-3
who = np.random.randn(1, hidden_size + 1) * 1e-3
ws = np.empty(3, dtype=object)
ws[:] = wie, weh, who #ws = np.array([wie, weh, who], dtype=object)
init_ws = lambda: ws - ws
optimizer = Momentum(init_ws, friction, lr)
tr_loss, val_acc = {}, {}

#for i in itertools.count():
i = 0
xs, ts = task.next_batch()
w = who
def f(w):
    embed, cache_embed = embed_forward(xs, wie)
    a, cache_act_fn = act_fn_forward(embed, weh, relu)
    s, cache_act_fn_2 = act_fn_forward(a, who, sigmoid)
    tr_loss[i], cache_ce = ce_forward(s, ts)
    return tr_loss[i], cache_embed, cache_act_fn, cache_act_fn_2, cache_ce

grad = eval_numerical_gradient(f, w)

_, cache_embed, cache_act_fn, cache_act_fn_2, cache_ce = f(w)
ds = ce_backward(cache_ce)
da, dwho = act_fn_backward(ds, sigmoid_prime, cache_act_fn_2)
dx, dweh = act_fn_backward(da, relu_prime, cache_act_fn)
dwie, dembed = embed_backward(dx, cache_embed)

print(grad)
print(dwho)
print(rel_difference(grad, dwho))

'''
ws = optimizer.update(ws, [dwie, dweh, dwho])

    val_xs, val_ts = task.val_data()
    embed, _ = embed_forward(val_xs, wie)
    a, _ = act_fn_forward(embed, weh, relu)
    s, _ = act_fn_forward(a, who, sigmoid)
    val_acc[i] = ((s > .5) == ts).mean()

    print(tr_loss[i])
    plot(tr_loss)

Some thigns take time, went out .172 at iteration 1000.
.181 (tanh, no hidden layers, SGD, 1e-3 wi, 1e-2 lr.)
best acc shofar: .186
comprar noise cancelling headphones
try with and without weight-tight
is a bias necessary in the embeddings?
see what's inside dembed
'''
