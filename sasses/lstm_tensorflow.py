import tensorflow as tf
import itertools
import sys
sys.path.append('../')
from utils import *
from parsers.any_txt import Task

learning_rate = 1e-2
beta1 = 0.9
beta2 = 0.999
num_layers = 5

batch_size = 200
embed_size = 50
seq_length = 25

task = Task(seq_length, batch_size)
x = tf.placeholder(tf.int32, shape=(None, None))
t = tf.placeholder(tf.float32, shape=(batch_size, seq_length, task.vocab_size))
batch_size_ph = tf.placeholder(tf.int32, [])
seq_length_ph = tf.placeholder(tf.int32, [])
# rnn_cell = tf.nn.rnn_cell.LSTMCell(embed_size)
# rnn = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * num_layers)#, state_is_tuple=True)
# initial_state = rnn.zero_state(batch_size_ph, tf.float32)
rnn = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers, embed_size)

wie = tf.Variable(tf.random_uniform([task.vocab_size, embed_size], -1.0, 1.0))
wro = tf.Variable(tf.random_normal([embed_size, task.vocab_size], stddev=0.01))
bro = tf.Variable(tf.constant(0.0, shape=(task.vocab_size,)))

embed = tf.nn.embedding_lookup(wie, x)
rnn_out = rnn(embed)
#rnn_out, final_state = tf.nn.dynamic_rnn(rnn, embed, initial_state=initial_state)#, time_major=True)
#rnn_out = batch, seq_length, embed
#wro = embed, vocab
# rnn_out = tf.reshape(rnn_out, (-1, rnn_out.get_shape()[2])) #TODO: Improve this
'''
y = tf.matmul(rnn_out, wro)
y = tf.reshape(y, (batch_size_ph, seq_length_ph, -1))
y = tf.nn.relu(y + bro)

y_softmax = tf.nn.softmax(y)
loss = tf.losses.softmax_cross_entropy(t, y)

optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2)
minimize = optimizer.minimize(loss)
'''
tr_loss, dev_loss, dev_acc = {}, {}, {}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in itertools.count():
        xs, ts = task.train_batch()
        ps(sess.run([rnn_out], feed_dict={x: xs, t: ts, batch_size_ph: batch_size, seq_length_ph: seq_length}))
        print('f')
        _, ys, tr_loss[i] = sess.run([minimize, y, loss], feed_dict={x: xs, t: ts, batch_size_ph: batch_size, seq_length_ph: seq_length})
        print ('Tr loss: ', tr_loss[i])

        if i % 10 == 0:
            xs, ts = task.dev_batch()
            ys, dev_loss[i] = sess.run([y_softmax, loss], feed_dict={x: xs, t: ts, batch_size_ph: batch_size, seq_length_ph: seq_length})
            dev_acc[i] = np.mean(np.argmax(ys, 2) == np.argmax(ts, 2))
            print ('Dev loss: ', str(dev_loss[i])

            xs = np.zeros((16,)).astype('int')
            xs[0] = np.random.randint(task.vocab_size)
            for i in range(15):
                xs_one = np.array([xs[i]]).reshape(1, 1)
                ys = sess.run(y_softmax, feed_dict={x: xs_one, batch_size_ph: 1, seq_length_ph: 1})[0][0]
                xs[i + 1] = np.random.choice(range(len(ys)), p=ys)
            print ('Sample: 0', " ".join(task.ixs_to_words(xs)))

        #plot(tr_loss)

        #print('\nAcc: {}. Loss: {}'.format(tr_acc[i], tr_loss[i]))
        #print ('Sample from training')
        #print (' '.join(task.ixs_to_words(xs[0, 20:25])))
        #top_ys = np.argpartition(ys[0], -5)[-5:]
        #for top_y in top_ys:
        #    print (' '.join(task.ixs_to_words(ys[0, 20:25, top_y])))

#TODO: add points and commas and enters to dataset.
#IDEA: take the word that is more deviated from the standard word distribution.
