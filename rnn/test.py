import numpy as np
from basic_layers import *
from utils import *

num_words = 10
input_size = 5
embed_size = 5
hidden_size = 5
batch_size = 6
xs = np.random.randn(num_words, input_size, batch_size)
ts = np.random.randint(0, hidden_size, size=(batch_size))
w_embed = np.random.randn(embed_size, input_size)
w = np.random.randn(hidden_size, embed_size * num_words + 1)

def f(w_embed):
    embed, cache_embed = embed_forward(xs, w_embed)
    a, cache_act_fn = act_fn_forward(embed, w, np.tanh)
    loss, p, cache_sm = softmax_forward(a, ts)
    return loss, cache_embed, cache_act_fn, cache_sm

grad = eval_numerical_gradient(f, w_embed)
loss, cache_embed, cache_act_fn, cache_sm = f(w_embed)

da = softmax_backward(cache_sm)
dx, dw = act_fn_backward(da, tanh_prime, cache_act_fn)
dw_embed, dembed = embed_backward(dx, cache_embed)

print (dw_embed)
print (grad)
print (rel_difference(dw_embed, grad))

'''
comprar noise cancelling headphones
try with and without weight-tight
is a bias necessary in the embeddings?
see what's inside dembed
'''
