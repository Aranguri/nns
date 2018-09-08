import time
import matplotlib.pyplot as plt
import numpy as np
from char_rnn_v5 import RNN
from utils import points_to_curve

models = [{'exp_name': 'large-training-2'}]#[{'update': 'adam', 'exp_name': 'adam-lr:1e-4v2', 'lr': 5e-3}]#, {'update': 'adam', 'exp_name': 'adam-lr:1e-3', 'lr': 1e-3}, {'update': 'adam', 'exp_name': 'adam-lr:1e-4', 'lr': 1e-4}]#, {'update': 'sgd', 'exp_name': 'sgd4'}]#,

for model in models:
    model['rnn'] = RNN(mode='sample-save-val-train', running_times=300000, **model)
    model['rnn'].run()

for model in models:
    train = model['rnn'].train_loss
    val = model['rnn'].val_loss
    plt.plot(points_to_curve(train), label='train')
    plt.plot(points_to_curve(val), label=model['exp_name'])

plt.legend(loc='best')
plt.show(block=False)
time.sleep(1e4)
