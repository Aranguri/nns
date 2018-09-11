import itertools
import numpy as np
from utils import *
from layers import *
from task import TextEnv
from plotter import plot

def run():
    hidden_size = 3
    batch_size = 2
    seq_length = 4
    running_times = 1000000
    learning_rate = 1e-3
    b1, b2 = .9, .999
    eps = 1e-6

    tenv = TextEnv(seq_length, batch_size)
    vocab_size = tenv.vocab_size
    np.random.seed(1)
    wxh = np.random.randn(4*hidden_size, vocab_size) * 1e-2
    whh = np.random.randn(4*hidden_size, hidden_size + 1) * 1e-2
    why = np.random.randn(vocab_size, hidden_size + 1) * 1e-2
    init_hc = lambda: (np.zeros((hidden_size, batch_size)), np.zeros((hidden_size, batch_size)))
    init_ws = lambda: (np.zeros_like(wxh), np.zeros_like(whh), np.zeros_like(why))
    ws, mws, vws = [wxh, whh, why], init_ws(), init_ws()
    loss_history = {}

    xs, ys = tenv.next_batch()

    def forward(wxh):
        return forward_lstm(xs, ys, wxh, whh, why, init_hc)

    def backward(cache):
        return backward_lstm(ys, cache, init_hc, init_ws)

    grad = eval_numerical_gradient(forward, wxh)
    print ('Numerical: ', grad)
    s, cache = forward(wxh)
    (dwxh, dwhh, dwhy), loss = backward(cache)
    print ('Analytical: ', dwxh)
    print ('Diff', rel_difference(grad, dwxh))

    '''
    for i in range(1, running_times):
        xs, ys = tenv.next_batch()
        s, cache = forward_lstm(xs, wxh, whh, why, init_hc)
        dws, loss_history[i] = backward_lstm(ys, cache, init_hc, init_ws)

        for w, dw, mw, vw in zip(ws, dws, mws, vws):
            # mw = b1 * mw + (1 - b1) * dw
            # mw /= (1 - b1 ** i)
            # vw = b2 * vw + (1 - b2) * dw ** 2
            # vw /= (1 - b2 ** i)
            w -= learning_rate * dw#mw / (np.sqrt(vw) + eps)

        if (i + 1) % 10000 == 0: plot(loss_history)

        if i % 100 == 0:
            scores = lstm_sample(tenv.rand_x(), wxh, whh, why, hidden_size, vocab_size)
            text = ''.join([tenv.i_to_char[s] for s in scores])
            print ('\n-------\nLoss: {:.2f} It: {}\n{}\n------\n'.format(loss_history[i], i, text))
    return loss_history
    '''

run()
