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
    lim = 2 ** (1 + int(np.log2(len(array))))
    plt.xlim(0, lim)
    plt.ylim(0, .6)
    plt.plot(array)
    plt.pause(1e-3)
