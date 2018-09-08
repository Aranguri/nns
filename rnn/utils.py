import itertools
import numpy as np

def cost(t, y):
    return sum((t - y) ** 2) / 2

relu = lambda x: np.maximum(0, x)
relu_prime = lambda x: (x > 0) * x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return (1 - sigmoid(x)) * sigmoid(x)

def tanh_prime(x):
    return 1 - np.square(np.tanh(x))

def expand(array):
    return np.expand_dims(array, axis=1)

def add_bias(array, axis=0):
    pad = ((0, 1), (0, 0)) if axis == 0 else ((0, 0), (0, 1))
    return np.pad(array, pad, 'constant', constant_values=1)#e-3)

def remove_bias(array, axis=0):
    return array[:-1] if axis == 0 else array[:, :-1]

def psh(arrays):
    for array in arrays: print (np.shape(array))

def normalize(array):
    array = array[:, 0]
    array = array - np.min(array)
    return array / np.sum(array)

def one_hot(pos, length):
    array = np.zeros((length))
    array[pos] = -1
    return array

def one_of_k(pos, length):
    array = np.zeros((length))
    array[pos] = 1
    return array

def points_to_curve(y):
    if type(y) == dict:
        y = [v for v in y.values()]
    #curve = [np.mean(points[max(0, i-10):i+10]) for i in range(len(points))]
    x = range(len(y))
    curve = np.poly1d(np.polyfit(x, y, 8))(np.unique(x[:-15]))
    return curve

def eval_numerical_gradient(f, x, h=1e-4):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])#, op_flags=['readwrite'])

    while not it.finished:
        i = it.multi_index

        old_xi = x[i]
        x[i] = old_xi + h
        pos = f(x)[0]
        x[i] = old_xi - h
        neg = f(x)[0]
        x[i] = old_xi

        grad[i] = np.sum((pos - neg)) / (2 * h)
        it.iternext()
    return grad

def rel_difference(a1, a2):
    #return np.sum(np.abs(a1 - a2))
    return np.max(np.abs(a1 - a2) / (np.maximum(np.abs(a1) + np.abs(a2), 1e-8)))
