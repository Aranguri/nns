# Do
try with and without weight-tight between embeddings
is a bias necessary in the embeddings?
see what's inside dembed
maybe calculating the ce and then the sigmoid has numerical problems

# Logs
Best acc shfoar: .85
Maybe the gradient calculation has some numerical problems. In particular, the gradient caclulated on the bias of weh numerically gives a different value than the analytical. I think that the analytical is the correct. The error appears when I multiply the weights by 1e-3
Some thigns take time, went out .172 at iteration 1000.
.181 (tanh, no hidden layers, SGD, 1e-3 wi, 1e-2 lr.)
best acc shofar: .186

# Others
comprar noise cancelling headphones
