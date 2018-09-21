import itertools
from time import time
from utils import *
from layers import *
from optimizer import *
from plotter import plot
from task import Task

hidden_size = 50
batch_size = 50
seq_length = 8
num_layers = 2
exp_name = 'none-v1'

task = Task(seq_length, batch_size)
ws = np.empty(2, dtype=object)
whs = np.empty(num_layers, dtype=object)
whs[0] = np.random.randn(4 * hidden_size, task.vocab_size + hidden_size + 1)
whs[1:] = [np.random.randn(4 * hidden_size, 2 * hidden_size + 1)] * (num_layers - 1)
wy = np.random.randn(task.vocab_size, hidden_size + 1)
ws[:] = whs, wy
ws = ws * 1e-3
#ws[1] = wy

init_whs = lambda: whs - whs #workaround: alternative to np.zeros_like(ws)
#init_wy = lambda: np.zeros_like(wy) #workaround: alternative to np.zeros_like(ws)
init_ws = lambda: ws - ws#(init_whs(), init_wy())
init_hc = lambda n=batch_size: np.zeros((2, hidden_size, n))
init_hscs = lambda n=batch_size: np.zeros((2, num_layers, hidden_size, n))
optimizer = Adagrad(init_ws)
tr_loss, val_acc, n = {}, {}, 0
#ws, tr_loss, val_acc, n = restore(exp_name)

for i in itertools.count(n):
    xs, ys = task.next_batch()
    tr_loss[i], caches = lstm_forward(xs, ys, ws, init_hscs)
    dws = lstm_backward(caches, init_hscs, init_whs)
    ws = optimizer.update(ws, dws)

    if (i + 1) % 10 == 0:
        xs, ys = task.get_val_data()
        val_acc[i] = lstm_val(xs, ys, ws, init_hscs, task)
        text = lstm_sample(task.rand_x(), ws, init_hscs, task.vocab_size, task)
        print (f'Loss: {dict_mean(tr_loss)} \n {task.array_to_sen(text)}\n\n')
        print (f'Val acc: {val_acc[i]} Max: {dict_max(val_acc)}')
        plot(val_acc)
        save(exp_name, ws, tr_loss, val_acc, i)
