import tensorflow as tf
import sys
sys.path.append('../')
from utils import *

batch_size = 2
input_size = 1
output_size = 1
T = 5

x = tf.placeholder(tf.float32, shape=(batch_size, T, input_size))
y = tf.placeholder(tf.float32, shape=(batch_size, T, 1))
rnn = tf.nn.rnn_cell.BasicRNNCell(output_size)
init_state = rnn.zero_state(batch_size, tf.float32)
output, final_state = tf.nn.dynamic_rnn(rnn, x, initial_state=init_state)
losses = tf.losses.mean_squared_error(y, output)
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(losses)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while True:
        xs = np.random.randint(10, size=(batch_size, T, input_size)) * 1e-2
        ys = [xs[:, :i+1, :].sum(axis=(1, 2)) for i in range(xs.shape[1])]
        ys = np.array(ys).T.reshape(batch_size, T, 1)
        sess.run(minimize, feed_dict={x: xs, y: ys})
        print(sess.run(output, feed_dict={x: xs, y: ys}))
        print(ys)
        print('\n')
        print(sess.run(losses, feed_dict={x: xs, y: ys}))

        #print(sess.run(losses, feed_dict={x: xs, y: ys}))
