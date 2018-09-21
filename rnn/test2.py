import numpy as np
from basic_layers import *
from utils import *

num_words = 20
input_size = 30
embed_size = 10
batch_size = 10
xs = np.random.randn(num_words, input_size, batch_size)
w_embed = np.random.randn(embed_size, input_size)
dout = np.random.randn(embed_size * num_words, batch_size)

def f(w_embed):
    embed, cache_embed = embed_forward(xs, w_embed)
    return embed, cache_embed

grad = eval_numerical_gradient(f, w_embed, dout)
loss, cache_embed = f(w_embed)
dw_embed, dembed = embed_backward(dout, cache_embed)

print (dw_embed)
print (grad)
print (rel_difference(dw_embed, grad))
