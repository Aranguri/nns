from task3 import Task
from os import system
import numpy as np
task = Task(seq_length=25)
score = []

for i in range(20):
    x, t = task.next_batch()
    x = task.ixs_to_words(x)
    t = task.ixs_to_words(t)
    print (' '.join(list(x)))
    next_word = input('')
    score.append(next_word == list(t)[-1])
    system('clear')
'''
10 => .2
20 => .3
20 => .3
20 => .25
(.3 + .3 + .25 + .1) / 3.5
.27
'''
print (np.mean(score))
