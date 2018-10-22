import sys
sys.path.append('/home/aranguri/Desktop/dev/nns/rnn')
from utils import *
from plotter import *
from task import Task
import numpy as np
import itertools

learning_rate = 1e-2
momentum = 0.9
words_size = 3
embed_size = 50
hidden_size = 200
task = Task()
sizes = [embed_size * words_size, hidden_size, task.vocab_size]
ws_embed = np.array([np.zeros((task.vocab_size, embed_size)) for _ in range(words_size)]) * 1e-3
ws = np.array([np.zeros((m + 1, n)) for m, n in zip(sizes, sizes[1:])]) * 1e-3
train_costs = {}
vws, vws_embed = np.zeros_like(ws), np.zeros_like(ws_embed)

def forward(xs, ts):
    embed = np.concatenate([x.dot(w) for x, w in zip(xs, ws_embed)], 1)
    zs = {0: embed}
    for i, w in enumerate(ws):
        zs[i + 1] = add_bias(zs[i], 1).dot(w)
    shift = zs[2].max()
    softmax_num = np.exp(zs[2] - shift)
    softmax = softmax_num.T / (softmax_num.sum(1) + 1e-10)
    costs = np.maximum(softmax[ts, np.arange(len(ts))], 1e-10) #We replace 0 with small values
    cost = np.sum(-np.log(costs))
    return cost, (xs, zs, softmax)

def backward(ts, cache):
    xs, zs, softmax = cache
    ds = softmax
    ds[ts, np.arange(len(ts))] -= 1
    dws, dws_embed, dzs = np.zeros_like(ws), np.zeros_like(ws_embed), {2: ds}

    for i in reversed(range(2)):
        dws[i] = dzs[i + 1].dot(add_bias(zs[i], 1)).T
        dzs[i] = remove_bias(ws[i]).dot(dzs[i + 1])

    dzs_embed = np.split(dzs[0], 3)
    for i, (dz, x) in enumerate(zip(dzs_embed, xs)):
        dws_embed[i] = dz.dot(x).T

    return dws, dws_embed

def val_net(xs, ts):
    embed = np.concatenate([x.dot(w) for x, w in zip(xs, ws_embed)], 1)
    zs = {0: embed}
    for i, w in enumerate(ws):
        zs[i + 1] = add_bias(zs[i], 1).dot(w)
    shift = zs[2].max()
    softmax_num = np.exp(zs[2] - shift)
    softmax = softmax_num.T / (softmax_num.sum(1) + 1e-10)
    for i, x in enumerate(xs.swapaxes(0, 1)):
        for word in np.argmax(x, 1):
            print(task.words[word][0], end=' ')
        print (task.words[np.argmax(softmax[:, i])][0])

    return np.mean(np.argmax(softmax, 0) == ts)

for i in itertools.count():
    xs, ts = task.next_batch()[0:2]
    train_costs[i], cache = forward(xs, ts)
    dws, dws_embed = backward(ts, cache)
    vws = momentum * vws + learning_rate * dws
    ws -= vws
    vws_embed = momentum * vws_embed + dws_embed
    ws_embed -= learning_rate * vws_embed
    if i % 1000 == 0:
        val_xs, val_ts = task.val_data()
        # plot(train_costs)
        print(f'Tr loss {dict_mean(train_costs)}')
        print(f'Val acc {val_net(val_xs, val_ts)}')


'''
Next thing: conv net from scratch
'''
