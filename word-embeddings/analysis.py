import sys
sys.path.append('../')
from utils import *
from task2 import Task

w = restore('lt-3000')[2]
task = Task(1)
unique_words = np.array([w[0] for w in task.words if w != None])

def v_by_w(w1):
    i = np.where(unique_words==w1)
    return w.dot(one_of_k(i, len(unique_words) + 1))

def distance(w1, w2):
    v = v_by_w(w1)
    u = v_by_w(w2)
    print (f'Distance between {w1} and {w2}: {np.square(u - v).sum()}')

def nearest_vectors(w1, v=None, k=10):
    v = v_by_w(w1) if type(v) == type(None) else v
    dist = np.zeros((len(unique_words)))

    for i, w2 in enumerate(unique_words):
        u = v_by_w(w2)
        dist[i] = np.square(u - v).sum()

    ix = sorted(np.arange(len(unique_words)), key=lambda i: dist[i])

    print (f'Vectors near {w1}')

    for i in ix[:k]:
        print (unique_words[i], dist[i])

def print_words():
    print (sorted(unique_words))

#print_words()
v = v_by_w('he')#v_by_w('he') - v_by_w('she') + v_by_w('him')
#v = - v_by_w('days') + v_by_w('day') + v_by_w('companies')# - v_by_w('two')# + v_by_w('he')
nearest_vectors(None, v)

#nearest_vectors('the')
#distance('he', 'she')
#distance('he', 'percent')
#distance('she', 'percent')
