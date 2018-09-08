Next steps:
I'm debugging the code to see what's producing the rel error of 1e-3 in the forward and backward functions. I tested lstm_step_forward, forward_affine, and softmax. I was doing something with the h_prev and c_prev. It seems that the error is related to the hs and cs

See whether layers.py works by using eval_numerical_gradient (maybe sampling is working bad because the loss decreases)
Plot loss history and valacc history to see whether it's working
Order what's here and define next features to implement. maybe dropout

https://arxiv.org/pdf/1808.09352.pdf

add graphs for loss history and check whether the features are working. Visualize how it works, (as it was done in karpathy's paper)
dropout (https://arxiv.org/pdf/1409.2329.pdf)
more layers
fix the fact that h and c aren't propagated through different seq_lengths
gru
calculate manually the weights to detect whether we are inside a quote or not.
clip gradients
l2 regularization
see how flexible this code is: if I remove bias or remove something in backprop, does the AI stop working?
Do:
* http://www.arxiv-sanity.com/1506.02078v2
* Cite 28 there
* map with all the possible thigns to do to a nn

Ideas
* one way to solve the truncated backprop thing: use the gradients of the 'pasada' t-1, they are old, but they may have imp information.
* similar papers to x-paper
* Alternative to 1-of-k vectors, so that characters are more distributed through the input.
* can we draw exactly a set of points (say 12) without consciously counting them?
* a neural network doesn't know what happens when you are in test time, because it doesn't learn from there. what if it learns? Like having 'pruebas simulacro.'
* Programming:
h, c = np.zeros((hidden_size)), np.zeros((hidden_size))
cache = {}
loss = 0
The prev three lines, in the future, should be reduced to
h, c, cache, loss = init_zero #or something like that. The compiler should detect what initial type they need to be.
They need to be initialized because I use +=.


grad = eval_numerical_gradient(forward, why)
print ('Numerical: ', grad)
, cache = forward(why)
dwxh, dwhh, dwhy = backward(xs, ys, cache)
print ('Analytical: ', dwhy)
print ('Diff', rel_difference(grad, dwhy))


Add adagrad or something that improves over sgd.
check if the clip thing in min-char-rnn is useful
chat
questions


Las perlitas:
--
e
Iacricnse roing ereands
p poierrs dring endands
I procrastinate doing errands
I prico errarostinas
---
Things
Wouldn't it be nice to have?
loss.append, x, y = some_func()
That is, to append to an array with only one line (as in the example from above)


'''next steps
why all the next letters are near the end in the dictionary?
stop adding features and check whethyer it's working. It seems it isn't working, cause even though it reduces the cost by a factor of 30%, the quality of the text is the same. It doesn't have the desired behavior
regularization
more layers
store nns

⅛
'''



Training log
adam, clip, lr:3e-4, b1=.999, b2=.9, 10k it => vacc=.5, vloss=3  (no 'rebote' just a plateau)
adam, clip, lr:1e-4, b1=.999, b2=.9, 20k it  => vacc=.5, vloss=2.6
adam, clip, lr:1e-5, b1=.999, b2=.9, 20k it  => vacc=, vloss=
