import time
import bitarray
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_prime(Z):
    return (1 - sigmoid(Z)) * sigmoid(Z)

def string_to_bits(string, size=None):
    if size: string = string.ljust(int(size / 8))
    ba = bitarray.bitarray()
    ba.frombytes(string.encode('utf-8'))
    array = np.array(ba, dtype=int)
    return array.reshape(len(array), 1)

def normalize(probs):
    probs = np.squeeze(probs)
    return probs / np.sum(probs)

def expand(array):
    return np.expand_dims(array, axis=1)

def bits_to_string(bits):
    bits = bits.reshape(int(len(bits) / 8), 8)
    sentence = ''
    for bin in bits:
        bin = [str(int(b)) for b in bin]
        sentence += chr(int(''.join(bin), 2))
    return sentence

def psh(arrays):
    for array in arrays: print (np.shape(array))

def mask_like(array, prob):
    mask = np.random.uniform(0, 1, array.shape)
    return np.array([[0 if v < prob else 1 for v in l] for l in mask])

#special utils for conv nets
def end_size(shape, input_size):
    neurons = input_size
    for type, size in shape:
        if type == 'conv': neurons -= size - 1
        elif type == 'pool': neurons /= size
        if int(neurons) - neurons != 0:
            raise ValueError('NN shape results in float end size amount of neurons')
    return int(neurons)

#*engineering department spawned*
def conv_mask(matrix):
    filter_size = matrix.shape[1] - matrix.shape[0] + 1
    values = [matrix.diagonal(i) for i in range(filter_size)]
    return np.sum(values, axis=1)

def scatter_plot(values):
    plt.scatter(np.arange(len(values)), values)
    plt.show(block=False)
    time.sleep(1000)
