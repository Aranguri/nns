#Harry 'tito' plotter
import matplotlib.pyplot as plt
import time
from utils import points_to_curve

def plot(array):
    if type(array) is dict: array = [v for v in array.values()]
    plt.plot(points_to_curve(array))
    plt.show()
