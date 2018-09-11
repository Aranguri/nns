import itertools
from time import time
from utils import *
from layers import *
from optimizer import *
from plotter import plot
from task import Task

hidden_size = 100
batch_size = 150
seq_length = 8
k = 3
exp_name = '>50 top1 i2'
itime = time()

task = Task(seq_length, batch_size)
ws = [np.random.randn(4 * hidden_size, task.vocab_size + hidden_size + 1) * 1e-3,
      np.random.randn(task.vocab_size, hidden_size + 1) * 1e-3]
init_hc = lambda n=batch_size: np.zeros((2, hidden_size, n))
init_ws = lambda: np.zeros_like(ws)
optimizer = Adam(init_ws)
tr_loss, val_acc, n = {}, {}, 0
ws, tr_loss, val_acc, n = restore(exp_name)

for i in itertools.count(n):
    xs, ys = task.next_batch()
    tr_loss[i], caches = lstm_forward(xs, ys, ws, init_hc, task)
    dws = lstm_backward(caches, init_hc, init_ws)
    ws = optimizer.update(ws, dws)

    if i % 100 == 0:
        xs, ys = task.get_val_data()
        val_acc[i] = lstm_val(xs, ys, ws, init_hc, k, task)
        #print (f'Val acc: {val_acc[i]/1000}. Time {round(time() - itime)}')
        #text = lstm_sample(task.rand_x(), ws, init_hc, task.vocab_size, task)
        #print (f'Loss: {dict_mean(tr_loss)} \n {task.array_to_sen(text)}\n\n')

    if (i + 1) % 300 == 0:
        save(exp_name, ws, tr_loss, val_acc, i)
        #plot(val_acc)
