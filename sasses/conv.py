import itertools
import sys
sys.path.append('../')
from conv_layers import *
from basic_layers import *
from utils import *
from task_load_docs import Task
from optimizer import *

learning_rate = 1e-1
friction = .9
batch_size = 100
reg = 1e-2

embed_len = 20
embed_depth = 20
layers = [(5, 7), (5, 8), (5, 9), (4, 10), (3, 11), (2, 12)]
wccs_sizes, wco_size = conv_structure([embed_len, embed_depth], layers)

wic = np.random.randn(embed_depth, 0)
wccs = np.array([np.random.randn(*wcc) * np.sqrt(2 / wcc[2]) * 5e-1 for wcc in wccs_sizes])
wco = np.random.randn(1, wco_size + 1) * np.sqrt(1 / (wco_size + 1))
ws = np.array([wic, wccs, wco], dtype=object)
init_ws = lambda: ws - ws
task = Task(embed_len, batch_size)
optimizer = Momentum(init_ws, learning_rate, friction)
tr_loss, tr_acc = [], []

for i in itertools.count():
    x, y = task.next_batch()
    wic = enlarged(wic, len(task.words) - wic.shape[1])

    a, cache_embed = embed_forward(x, wic)
    cache_convs = {}
    for j, wcc in enumerate(wccs):
        a, cache_convs[j] = conv_forward(a, wcc, relu)
        a -= a.mean()
    b, cache_fc = affine_fn_forward(a.reshape(-1, a.shape[2]), wco, sigmoid)
    loss_ce, cache_cost = ce_forward(b, y)
    loss_reg, cache_reg = regularization_forward([wic, wccs, wco], reg)
    tr_loss.append(loss_ce + loss_reg)

    db = ce_backward(cache_cost)
    da, dwco = affine_fn_backward(db, cache_fc, sigmoid_prime)
    da = da.reshape(*a.shape)
    dwccs = np.zeros_like(wccs)
    for j, cache_conv in enumerate(reversed(list(cache_convs.values()))):
        da, dwccs[j] = conv_backward(da, cache_conv, relu_prime)
    dx, dwic = embed_backward(da, cache_embed)
    dwccs = np.array([dwccs[j] for j in reversed(range(len(dwccs)))])
    dwic, dwccs, dwco = [dwic, dwccs, dwco] + regularization_backward(cache_reg)
    print(wic[0][0])

    wic, wccs, wco = optimizer.update([wic, wccs, wco],
                                      [dwic, dwccs, dwco])

    if i % 10 == 0:
        x, y = task.next_batch()
        wic = enlarged(wic, len(task.words) - wic.shape[1])

        a, cache_embed = embed_forward(x, wic)
        for wcc in wccs:
            a, _ = conv_forward(a, wcc, relu)
        d, cache_fc = affine_fn_forward(a.reshape(-1, a.shape[2]), wco, sigmoid)
        tr_acc.append([(d > .5) == y][0][0])
        correct = np.mean(tr_acc[-1])
        print (f'Batch: {correct}. NN: {int((d > .5)[0][0])}. reality: {y[0]}. value {d[0][0]}')#'. txt: {p[:75]}')

    print(f'Loss {np.mean(tr_loss)}. Acc: {np.mean(tr_acc)}')
    plot([np.mean(tr_loss[i:i+1]) for i in range(len(tr_loss) // 1)])
    #plot([np.mean(list(tr_loss.values())[:i + 1]) for i in range(len(tr_loss))])
    #plot([np.mean(tr_acc)[:i + 1]) for i in range(len(tr_loss))])
