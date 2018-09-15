import itertools
from time import time
from utils import *
from layers import *
from optimizer import *
from plotter import plot
from task import Task

hidden_size = 1
batch_size = 1
seq_length = 2
num_layers = 1
vocab_size = 2
exp_name = 'none-v1'
np.random.seed(1)

task = Task(seq_length, batch_size)
whs = np.empty(num_layers, dtype=object)
whs[0] = np.random.randn(4 * hidden_size, vocab_size + hidden_size + 1)
whs[1:] = [np.random.randn(4 * hidden_size, 2 * hidden_size + 1)] * (num_layers - 1)
wy = np.random.randn(vocab_size, hidden_size + 1)
#ws = np.array([whs, wy], dtype=object) * 1e-3
init_whs = lambda: whs - whs #workaround: alternative to np.zeros_like(ws)
init_wy = lambda: np.zeros_like(wy) #workaround: alternative to np.zeros_like(ws)
init_hc = lambda n=batch_size: np.zeros((2, hidden_size, n))
init_hscs_zero = lambda n=batch_size: np.zeros((2, num_layers, hidden_size, n))
def init_hscs(n=batch_size):
    np.random.seed(1)
    #print(np.random.randn())
    hs = np.random.randn(1, num_layers, hidden_size, n)
    cs = np.random.randn(1, num_layers, hidden_size, n)
    hscs = np.concatenate((hs, cs))
    return hscs
#optimizer = Adam(init_ws)
tr_loss, val_acc, n = {}, {}, 0
#ws, tr_loss, val_acc, n = restore(exp_name)

whs1 = np.random.randn(4 * hidden_size, vocab_size + hidden_size + 1)
whs2 = [np.random.randn(4 * hidden_size, 2 * hidden_size + 1)] * (num_layers - 1)
xs = np.random.randn(seq_length, vocab_size, batch_size)
ys = np.random.randint(0, vocab_size, size=(seq_length, batch_size))
def flstm(whs1):
    whs = np.empty(num_layers, dtype=object)
    whs[0] = whs1
    whs[1:] = whs2
    return lstm_forward(xs, ys, whs, wy, init_hscs_zero, task)
dws_num = eval_numerical_gradient(flstm, whs1)
loss, caches = flstm(whs1)
dwhs, dwy = lstm_backward(caches, init_hscs_zero, init_whs, init_wy)
print (dws_num)
print (dwhs[0])
ps (dws_num, dwhs[0])
print (rel_difference(dws_num, dwhs[0]))

'''
for i in itertools.count(n):
    xs, ys = task.next_batch()
    tr_loss[i], caches = lstm_forward(xs, ys, ws, init_hscs, task)
    dws = lstm_backward(caches, init_hscs, init_ws)
    #print (f'Loss: {dict_mean(tr_loss)}')
    ws = optimizer.update(ws, dws)

    if (i + 1) % 10 == 0:
        xs, ys = task.get_val_data()
        val_acc[i] = lstm_val(xs, ys, ws, init_hscs, task)
        print (f'Val acc: {val_acc[i]}')
        plot(tr_loss)
        #text = lstm_sample(task.rand_x(), ws, init_hc, task.vocab_size, task)
        #print (f'Loss: {dict_mean(tr_loss)} \n {task.array_to_sen(text)}\n\n')
        save(exp_name, ws, tr_loss, val_acc, i)
'''
