import itertools
from utils import *
from layers import *
from task import Task

learning_rate = 1e-3
hidden_size = 100
batch_size = 5
seq_length = 5
beta1 = .9
beta2 = .999

task = Task(seq_length, batch_size)
wxh = np.random.randn(4 * hidden_size, task.vocab_size) * 1e-2
whh = np.random.randn(4 * hidden_size, hidden_size + 1) * 1e-2
why = np.random.randn(task.vocab_size, hidden_size + 1) * 1e-2
init_hc = lambda n=batch_size: np.zeros((2, hidden_size, n))
init_ws = lambda: np.zeros_like([wxh, whh, why])
mws, vws = init_ws(), init_ws()
loss_history = {}

for i in itertools.count(1):
    xs, ys = task.next_batch()
    loss_history[i], caches = lstm_forward(xs, ys, wxh, whh, why, init_hc)
    dws = lstm_backward(caches, init_hc, init_ws)
    for w, dw, mw, vw in zip([wxh, whh, why], dws, mws, vws):
        #dw += reg * w
        mw = beta1 * mw + (1 - beta1) * dw
        vw = beta2 * vw + (1 - beta2) * dw ** 2
        mw = mw / (1 - beta1 ** (i + 1))
        vw = vw / (1 - beta2 ** (i + 1))
        w -= learning_rate * mw / (np.sqrt(vw) + 1e-8)

    print (dict_mean(loss_history))
    if i % 10000 == 0:
        text = lstm_sample(task.rand_x(), wxh, whh, why, init_hc, task.vocab_size, task)
        print (f'Loss: {dict_mean(loss_history)} \n {task.array_to_sen(text)}\n\n')
