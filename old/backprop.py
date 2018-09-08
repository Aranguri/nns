import numpy as np
import matplotlib.pyplot as plt
import time
learning_rate = 0.01

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_prime(Z):
    return (1 - sigmoid(Z)) * sigmoid(Z)

W1 = np.random.uniform(-1, 1, (2, 2))
W2 = np.random.uniform(-1, 1, 2)
U1 = np.array([[1, 1], [1, 1]])
U2 = np.array([1, 1])

costs = []

for i in range(10000):
    X1 = np.random.uniform(0, 1, 2)
    Y = sigmoid(np.dot(U2, sigmoid(np.dot(U1, X1))))
    Z1 = np.dot(W1, X1)
    X2 = sigmoid(Z1)
    Z2 = np.dot(W2, X2)
    X3 = sigmoid(Z2)
    cost = (Y - X3) ** 2 / 2
    #print (cost)
    dCdZ2 = (X3 - Y) * sigmoid_prime(Z2)
    dCdW2 = dCdZ2 * X2
    dZ2dZ1 = W2 * sigmoid_prime(Z1)
    dCdZ1 = dCdZ2 * dZ2dZ1
    print (np.shape(dCdZ1))
    print (np.shape(X1))
    dCdw11 = dCdZ1[0] * X1
    dCdw12 = dCdZ1[1] * X1

    W2 -= dCdW2
    W1[0] -= dCdw11
    W1[1] -= dCdw12

    costs.append(cost)
    print (np.average(costs))

print (W1, W2)
plt.plot(costs)
plt.show(block=False)
time.sleep(1000)
