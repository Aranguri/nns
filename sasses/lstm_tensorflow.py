import tensorflow as tf
import itertools
import sys
sys.path.append('../')
from utils import *
from parsers.any_txt import Task

batch_size = 200
embed_size = 50
seq_length = 25

task = Task(seq_length, batch_size)
x = tf.placeholder(tf.int32, shape=(None, None))
t = tf.placeholder(tf.float32, shape=(batch_size, seq_length, task.vocab_size))

embeddings = tf.Variable(tf.random_uniform([task.vocab_size, embed_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, x)

rnn = tf.nn.rnn_cell.LSTMCell(embed_size)
batch_size_ph = tf.placeholder(tf.int32, [])
seq_length_ph = tf.placeholder(tf.int32, [])
init_state = rnn.zero_state(batch_size_ph, tf.float32)
rnn_out, final_state = tf.nn.dynamic_rnn(rnn, embed, initial_state=init_state)
#w = embed, vocab
#x = batch, seq_length, embed
w1 = tf.Variable(tf.random_normal([embed_size, task.vocab_size], stddev=0.01))
b1 = tf.Variable(tf.constant(0.0, shape=(task.vocab_size,)))
rnn_out = tf.reshape(rnn_out, (-1, rnn_out.get_shape()[2]))
y = tf.matmul(rnn_out, w1)
y = tf.reshape(y, (batch_size_ph, seq_length_ph, -1))
y = tf.nn.relu(y + b1)

y_softmax = tf.nn.softmax(y)
loss = tf.losses.softmax_cross_entropy(t, y)
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(loss)

tr_loss, tr_acc = {}, {}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in itertools.count():
        xs, ts = task.next_batch()
        _, ys, tr_loss[i] = sess.run([minimize, y, loss], feed_dict={x: xs, t: ts, batch_size_ph: batch_size, seq_length_ph: seq_length})
        tr_acc[i] = np.mean(np.argmax(ys, 2) == np.argmax(ts, 2))
        print('Acc: {}. Loss: {}'.format(tr_acc[i], tr_loss[i]))

        if i % 5 == 0:
            xs = np.random.randint(task.vocab_size)
            for _ in range(15):
                xs = np.array([xs]).reshape(1, 1)
                ys = sess.run(y_softmax, feed_dict={x: xs, batch_size_ph: 1, seq_length_ph: 1})[0][0]
                ys = np.random.choice(range(len(ys)), p=ys)
                xs = ys
                print (task.ixs_to_words(ys), end=' ')
            print ('\n')
            plot(tr_acc)
