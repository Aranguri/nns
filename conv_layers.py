import numpy as np
from utils import *

#TODO: add batches, add strides, examine pytorch/tf, understand how matmul work and how it's different from a for 

'''
X = (LI, D, B)
W = (LF, D, N)
A = (LI - LF + 1, N, B)
'''
def conv_forward(x, ws, fn):
    input_len, _, batch_size = x.shape
    filter_len, filter_depth, filter_num = ws.shape
    new_len = input_len - filter_len + 1
    a, z = np.zeros((2, new_len, filter_num, batch_size))

    for i in range(filter_num):
        for j in range(new_len):
            w = ws[:, :, i]
            w = w.reshape(filter_len, filter_depth, 1)
            z[j][i] = (w * x[j:j + filter_len]).sum((0, 1))
            a[j][i] = fn(z[j][i])

    return a, (x, ws, z)

def conv_backward(dout, cache, fn_prime):
    x, ws, z = cache
    input_len, (filter_len, _, filter_num) = x.shape[0], ws.shape
    dws, dx = np.zeros_like(ws), np.zeros_like(x)

    for i in range(filter_num):
        for j in range(input_len - filter_len + 1):
            w = ws[:, :, i]
            dz = fn_prime(z[j][i]) * dout[j][i]
            dws[:, :, i] += x[j:j + filter_len].dot(dz)
            w = w.reshape(filter_len, -1, 1)
            dz = dz.reshape(1, 1, -1)
            dx[j:j + filter_len] += w * dz

    return dx, dws

def pool_forward(x, len_pool):
    len_input, depth_input, batch_size = x.shape
    z = np.zeros((len_input // len_pool, depth_input, batch_size))

    for i in range(len_input // len_pool):
        z[i] = np.amax(x[i*len_pool:(i+1)*len_pool], 0)

    return z, (z, x, len_pool)

def pool_backward(dout, cache):
    z, x, len_pool = cache
    len_input = x.shape[0]
    dx = np.zeros_like(x)

    for i in range(len_input // len_pool):
        x_portion = x[i*len_pool:(i+1)*len_pool]
        dx[i*len_pool:(i+1)*len_pool] = (x_portion == z[i]) * dout[i]

    return dx

def similarity_cost_forward(v1, v2, same):
    diff = (1/2 if same else -1/2) * np.sum(np.square(v1 - v2))
    return diff, (v1, v2, same)

def similarity_cost_backward(cache):
    v1, v2, same = cache
    dv1 = v1 - v2 if same else v2 - v1
    dv2 = v2 - v1 if same else v1 - v2
    return dv1, dv2

def grad_check():
    def f(x):
        a, cache_conv = conv_forward(x, ws, relu)
        b, cache_pool = pool_forward(a, 3)
        return b, cache_conv, cache_pool

    x = np.random.randn(17, 4, 6)
    ws = np.random.randn(2, 3, 4)
    b, cache_conv, cache_pool = f(x)
    dout = np.random.randn(*b.shape)
    da = pool_backward(dout, cache_pool)
    dws, dx = conv_backward(da, cache_conv, relu_prime)
    grad = eval_numerical_gradient(f, x, dout)
    print (dx)
    print (grad)
    print(rel_difference(dx, grad))
