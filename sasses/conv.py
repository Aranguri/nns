import itertools
import sys
sys.path.append('../')
from conv_layers import *
from basic_layers import *
from utils import *
from task_load_docs import Task

embed_depth = 8
hidden_depth = 8
hidden_len = 8
output_depth = 8
output_len = 4

batch_size = 1
seq_length = 16
learning_rate = 1e-1

task = Task(seq_length, batch_size)
w_embed = np.random.randn(embed_depth, 0) * 1e-2
ws1 = np.random.randn(hidden_len, embed_depth, hidden_depth) * 1e-2
ws2 = np.random.randn(output_len, hidden_depth, output_depth) * 1e-2
loss_acc = {}

for i in itertools.count():
    v, w, new_words, same = task.next_batch()
    w_embed = enlarged(w_embed, new_words)
    c, caches = [None, None], []
    dws1_acc, dws2_acc = 0, 0

    for j, x in enumerate([v, w]):
        embed, cache_embed = embed_forward(x, w_embed, keep_dims=True)
        b, cache_conv1 = conv_forward(embed, ws1, relu)
        c[j], cache_conv2 = conv_forward(b, ws2, identity)
        caches.append([cache_embed, cache_conv1, cache_conv2])

    loss_acc[i], cache_cost = similarity_cost_forward(c[0], c[1], same)
    #print (loss_acc[i], same)
    dc1, dc2 = similarity_cost_backward(cache_cost)

    for dc, cache in zip([dc1, dc2], caches):
        cache_embed, cache_conv1, cache_conv2 = cache
        dws2, db = conv_backward(dc, cache_conv2, identity_prime)
        dws1, dembed = conv_backward(db, cache_conv1, relu_prime)
        dw_embed, dx = embed_backward(dembed, cache_embed)
        dws1_acc += dws1
        dws2_acc += dws2

    #ws1 -= learning_rate * dws1_acc
    #ws2 -= learning_rate * dws2_acc
    print(list(loss_acc))
    if i > 1:
        print([np.mean(list(loss_acc)[:i]) for i in range(len(loss_acc) - 1)])
        plot([np.mean(list(loss_acc.values())[:i]) for i in range(1, len(loss_acc) - 1)])
