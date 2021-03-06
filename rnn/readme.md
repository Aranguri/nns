#Mission/goal
Build something that in test-mode acts in a conversational way, like a chat. (in the future, it can learn to perform actions besides talking. One of these actions could be looking in the internet for more information.) So, the idea is to create something alive. We can define a validation metric and improve it. The first step seems to be the structure for that problem (looking for the dataset, seeing how I can encode the question/message by the other person.) That seems the right problem, after that we can entirely focus on adding technical features.

#New features
Ordenar. Maybe what's failing is adam.

1) more layers. (next step: check the gradient to see if it's correct.)
2) l2 regularization
3) clip gradients
4) Visualize features as it was done in karpathy's paper
5) see how flexible this code is: if I remove bias or remove something in backprop, does the AI stop working?
Learning rate decay
dropout (https://arxiv.org/pdf/1409.2329.pdf)
can we fix the fact that h and c aren't propagated through different seq_lengths?

#Other projects
framework for automatic differentation
gru
chat
summarizer
questions
rl (do with rl what I did with backprop)
use evolution for the structure

#Implementation details
6) /home/aranguri/Desktop/dev/nns/rnn/layers.py:105: RuntimeWarning: invalid value encountered in true_divide
  p = exp_s / exp_s.sum(0)

#reading/learning
calculate manually the weights to detect whether we are inside a quote or not.
why gpus is faster? use gcloud

#To process
https://arxiv.org/pdf/1808.09352.pdf
Do:
* http://www.arxiv-sanity.com/1506.02078v2
* Cite 28 there
* map with all the possible thigns to do to a nn

#how to use eval_numerical_gradient
create a function that calls the desired function to evaluate. don't create random variables inside that function.

whs1 = np.random.randn(4 * hidden_size, vocab_size + hidden_size + 1)
whs2 = [np.random.randn(4 * hidden_size, 2 * hidden_size + 1)] * (num_layers - 1)
whs_added = np.random.randn(4 * hidden_size, 2 * hidden_size + 1)
pos = 8
xs = np.random.randn(seq_length, vocab_size, batch_size)
ys = np.random.randint(0, vocab_size, size=(seq_length, batch_size))

def flstm(whs_added):
    whs = np.empty(num_layers, dtype=object)
    whs[0] = whs1
    whs[1:] = whs2
    whs[pos] = whs_added
    return lstm_forward(xs, ys, whs, wy, init_hscs)

loss, caches = flstm(whs_added)
dwhs, dwy = lstm_backward(caches, init_hscs, init_whs)
dws_num = eval_numerical_gradient(flstm, whs_added)

print (dws_num, '\n', dwhs[pos], '\n', dws_num.shape, dwhs[pos].shape)
print (rel_difference(dws_num, dwhs[pos]))

#Ideas
* one way to solve the truncated backprop thing: use the gradients of the 'pasada' t-1, they are old, but they may have imp information.
* Alternative to 1-of-k vectors, so that characters are more distributed through the input.
* can we draw exactly a set of points (say 12) without consciously counting them?
* a neural network doesn't know what happens when you are in test time, because it doesn't learn from there. what if it learns? Like having 'pruebas simulacro.'
* Programming1
h, c = np.zeros((hidden_size)), np.zeros((hidden_size))
cache = {}
loss = 0
The prev three lines, in the future, should be reduced to
h, c, cache, loss = init_zero #or something like that. The compiler should detect what initial type they need to be.
They need to be initialized because I use +=.
* Programming2
Wouldn't it be nice to have?
loss.append, x, y = some_func()
That is, to append to an array with only one line (as in the example from above)

#Las perlitas:
e
Iacricnse roing ereands
p poierrs dring endands
I procrastinate doing errands
I prico errarostinas

#Training log
adam, clip, lr:3e-4, b1=.999, b2=.9, 10k it => vacc=.5, vloss=3  (no 'rebote' just a plateau)
adam, clip, lr:1e-4, b1=.999, b2=.9, 20k it  => vacc=.5, vloss=2.6
adam, clip, lr:1e-5, b1=.999, b2=.9, 20k it  => vacc=, vloss=

Tuning hyperparams .45 sec each
h=100, b=10, lr=3e-3, s=6 .58
h=100, b=20, lr=3e-3, s=6 .61
h=100, b=20, lr=3e-3, s=2 .5
h=100, b=20, lr=3e-3, s=4 .57
h=100, b=20, lr=3e-3, s=8 .63
h=100, b=20, lr=3e-3, s=10 .63
h=100, b=20, lr=3e-2, s=10 .6
h=100, b=20, lr=1e-3, s=10 .53
h=150, b=20, lr=3e-3, s=8 .56
h=150, b=20, lr=1e-2, s=8 .6
h=150, b=50, lr=1e-2, s=8 .67 @59
h=150, b=50, lr=3e-2, s=8 .63
h=150, b=100, lr=3e-2, s=8 .64 (.71 @700)

\ >.5 @60
h=100, b=10, lr=1e-3, s=4 .15
h=100, b=10, lr=3e-3, s=4 .13
h=100, b=35, lr=3e-3, s=4 .17
h=100, b=60, lr=3e-3, s=4 .17 (.2 @85)
h=100, b=60, lr=3e-3, s=4 .18 (.2 @82)
h=100, b=60, lr=1e-2, s=4 .2 (.21 @68)
h=100, b=200, lr=3e-2, s=4
h=100, b=75, lr=3e-2, s=4

#Visualizing activations
##2: after a
green for f, l, r, g
red for n, t, s
it also worked for uppercase A
##4
green for m starting sentences
red for s (often s that are in the end of a sentence)
##5
red for some punctuation marks .!?
##6
green: last letters
##8
very green: on ,:;!.
##10
it detects ends of the words: but, to, not, by, and, go, do, of, what, that (it detects To and to, and But and but. It fails in doubt vs do and gone vs go)
##14
Red with \n
Green with the first letter of a sentence

# Benchmarks
Input 2, hidden_size = 100, batch_size = 50, seq_length = 8: peaked at .33
