import sys
sys.path.append('/home/aranguri/Desktop/dev/nns/rnn')
from utils import *
from plotter import *
from task import Task
import numpy as np
import itertools

learning_rate = 3e-2
embed_size = 50
hidden_size = 200
task = Task()
sizes = [task.vocab_size * 3, hidden_size, task.vocab_size]
ws_embed = [np.random.randn(task.vocab_size, embed_size) for _ in range(3)]
ws = np.array([np.random.randn(m, n + 1) for m, n in zip(sizes[1:], sizes)]) * 1e-3
train_costs = {}

def forward(xs, ts):
    zs = {0: xs}
    test = np.concatenate([w.dot(x) for x, w in xs, ws_embed])

    for i, w in enumerate(ws):
        zs[i + 1] = w.dot(add_bias(zs[i]))
    softmax_num = np.exp(zs[2])
    softmax = softmax_num / softmax_num.sum(0)
    cost = np.sum(-np.log(softmax[ts, np.arange(len(ts))]))
    return cost, (zs, softmax)

def backward(ts, cache):
    zs, softmax = cache
    ds = softmax
    ds[ts, np.arange(len(ts))] -= 1
    dws, dzs = np.zeros_like(ws), {2: ds}

    for i in reversed(range(2)):
        dws[i] = dzs[i + 1].dot(add_bias(zs[i]).T)
        dzs[i] = remove_bias(ws[i].T).dot(dzs[i + 1])

    return dws

for i in itertools.count():
    xs, ts = task.next_batch()[0:2]
    train_costs[i], cache = forward(xs, ts)
    dws = backward(ts, cache)
    ws -= learning_rate * dws
    plot(train_costs)
    print(dict_mean(train_costs))
