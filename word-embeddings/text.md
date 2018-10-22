# Do
try with and without weight-tight between embeddings
is a bias necessary in the embeddings?
see what's inside dembed
maybe calculating the ce and then the sigmoid has numerical problems
easier way to perform eval_numerical_gradient. maybe: individual eval for each fn in basic_layers

# Logs
Best val error shofar: 2.6 (epoch 8)
using the same parametrs as hinton-nn assign2 achieves the same performance. Try momentum
Best acc shfoar: .92
Maybe the gradient calculation has some numerical problems. In particular, the gradient caclulated on the bias of weh numerically gives a different value than the analytical. I think that the analytical is the correct. The error appears when I multiply the weights by 1e-3
Some thigns take time, went out .172 at iteration 1000.
.181 (tanh, no hidden layers, SGD, 1e-3 wi, 1e-2 lr.)
best acc shofar: .186

# Others
comprar noise cancelling headphones

# eval_numerical_gradient
# x = np.random.randn(task.num_words, task.vocab_size, batch_size)
# t = np.random.randint(task.vocab_size, size=(batch_size,))
x, t, new_epoch = task.next_batch()

def f(wie):
    embed, cache_embed = embed_forward(x, wie)
    s, cache_act_fn = act_fn_forward(embed, weh, sigmoid)
    u, cache_act_fn_2 = act_fn_forward(s, who, identity)
    p = softmax_forward(u)
    loss, cache_ce = ce_forward(p, t)
    return loss, cache_ce, cache_act_fn_2, cache_act_fn, cache_embed

grad = eval_numerical_gradient(f, wie)
loss, cache_ce, cache_act_fn_2, cache_act_fn, cache_embed = f(wie)
du = softmax_ce_backward(cache_ce)
ds, dwho = act_fn_backward(du, identity_prime, cache_act_fn_2)
da, dweh = act_fn_backward(ds, sigmoid_prime, cache_act_fn)
dwie, dembed = embed_backward(da, cache_embed)
print (grad)
print (dwie)
print(rel_difference(grad, dwie))
# ws = optimizer.update(ws, [dwie, dweh, dwho])
