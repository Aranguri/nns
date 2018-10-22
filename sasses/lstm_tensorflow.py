import tensorflow as tf
import itertools
import sys
sys.path.append('../')
from utils import *
from parsers.any_txt import Task

batch_size = 80
seq_length = 15

task = Task(seq_length, batch_size)
x = tf.placeholder(tf.float32, shape=(None, None, task.vocab_size))
t = tf.placeholder(tf.float32, shape=(batch_size, seq_length, task.vocab_size))
rnn = tf.nn.rnn_cell.LSTMCell(task.vocab_size)
batch_size_ph = tf.placeholder(tf.int32, [])
init_state = rnn.zero_state(batch_size_ph, tf.float32)
y, final_state = tf.nn.dynamic_rnn(rnn, x, initial_state=init_state)
y_softmax = tf.nn.softmax(y)
loss = tf.losses.softmax_cross_entropy(t, y)
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in itertools.count():
        xs, ts = task.next_batch()
        _, ys, losses = sess.run([minimize, y, loss], feed_dict={x: xs, t: ts, batch_size_ph: batch_size})
        acc = np.mean(np.argmax(ys, 2) == np.argmax(ts, 2))
        print(f'Acc: {acc}. Loss: {losses}')

        if i % 5 == 0:
            xs = np.random.randint(task.vocab_size)
            for _ in range(15):
                xs = one_of_k(xs, task.vocab_size).reshape(1, 1, task.vocab_size)
                ys = sess.run(y_softmax, feed_dict={x: xs, batch_size_ph: 1})[0][0]
                ys = np.random.choice(range(len(ys)), p=ys)
                xs = ys
                print (task.ixs_to_words(ys), end=' ')
            print ('\n')
