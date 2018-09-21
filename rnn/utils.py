import pickle
import itertools
import numpy as np
import colorful

relu = lambda x: np.maximum(0, x)
relu_prime = lambda x: (x > 0) * x
sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_prime = lambda x: (1 - sigmoid(x)) * sigmoid(x)
tanh_prime = lambda x: 1 - np.square(np.tanh(x))

def expand(array):
    return np.expand_dims(array, axis=1)

def add_bias(array, axis=0):
    pad = ((0, 1), (0, 0)) if axis == 0 else ((0, 0), (0, 1))
    return np.pad(array, pad, 'constant', constant_values=1)

def remove_bias(array, axis=0):
    return array[:-1] if axis == 0 else array[:, :-1]

def ps(a1, a2=None, a3=None, a4=None, a5=None):
    for a in [a1, a2, a3, a4, a5]:
        if a is not None:
            print (np.shape(a))

def normalize(array):
    array = array[:, 0]
    array = array - np.min(array)
    return array / np.sum(array)

def one_of_k(pos, length):
    if type(pos) == list or type(pos) == np.ndarray:
        array = np.zeros((length * len(pos)))
        for i, p in enumerate(pos):
            array[i * length + p] = 1
    else:
        array = np.zeros((length))
        array[pos] = 1
    return array

def points_to_curve(y):
    x = range(len(y))
    curve = np.poly1d(np.polyfit(x, y, 8))(np.unique(x))
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
    return np.max(np.abs(a1 - a2) / (np.maximum.reduce([np.abs(a1), np.abs(a2), np.ones_like(a2) * 1e-15])))

def dict_max(d):
    return np.max(list(d.values()))

def dict_mean(d):
    return np.mean(list(d.values()))

def dict_sum(d):
    values = np.array([v for v in d.values()])
    return values.sum(0)

def save(name, *args):
    with open(f'savings/{name}.pkl', 'wb') as f:
        pickle.dump(args, f)

def restore(name):
    with open(f'savings/{name}.pkl', 'rb') as f:
        print ('net restored')
        return pickle.load(f)

def init_cprint():
    colors = {'black': (0, 0, 0)}
    for i in range(0, 256):
        colors[str(i)] = (i, 255, i)
        colors[str(i + 256)] = (255, 255-i, 255-i)
        colorful.use_palette(colors)
        colorful.update_palette(colors)

def cprint(char, color):
    color = (abs(color) ** (1/3)) * (color / abs(color)) + 1
    color = str(int(color * 255))
    print (getattr(colorful, f'black_on_{color}')(char), end='')
