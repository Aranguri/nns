from nn import NN
import matplotlib.pyplot as plt
import time

accuracies = []

for lr in [.01, .03, .1, .3, .1]:
    nn = NN()
    nn.learning_rate = lr
    accuracies.append(nn.run())
    print (accuracies)

plt.plot(accuracies)
plt.show(block=False)
time.sleep(1000)
