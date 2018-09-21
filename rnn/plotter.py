#Harry 'tito' plotter
import matplotlib.pyplot as plt
import time
import numpy as np
from utils import points_to_curve
plt.ion()

def plot(array):
    plt.cla()
    if type(array) is dict:
        array = [v for v in array.values()]
    xlim = 2 ** (1 + int(np.log2(len(array))))
    ylim = 2 ** (1 + int(np.log2(np.maximum(max(array), 1e-8))))

    plt.xlim(0, xlim)
    plt.ylim(0, ylim)#2000)#.6)
    plt.plot(array)
    plt.pause(1e-8)
