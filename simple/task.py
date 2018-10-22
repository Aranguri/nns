import numpy as np
import matplotlib.pyplot as plt

class Task:
	def __init__(self):
		train_size = 100
		self.x1 = np.random.randn(train_size, 2)
		self.x2 = np.random.randn(train_size, 2)
		self.x2 += 3 * np.sign(self.x2)
		self.x = np.concatenate((self.x1, self.x2))
		self.y = np.concatenate((np.ones(100), np.zeros(100)))

	def visualize(self):
		for x, y in zip(self.x, self.y):
			color = 'red' if y == 1 else 'blue'
			plt.scatter(x[0], x[1], color=color)
		plt.plot()
		plt.pause(100)

Task().visualize()
