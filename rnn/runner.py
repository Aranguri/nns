import time 
from utils import points_to_curve
import matplotlib.pyplot as plt

lrs = (1e-2, 1e-3, 1e-4)
for lr in lrs:
    from LSTM import run
    loss_history = run(lr)
    plt.plot(points_to_curve(loss_history), label=lr)

plt.legend(loc='best')
plt.show(block=False)
time.sleep(1e5)
