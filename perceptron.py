import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

class Perceptron:
	def __init__(self, inputs_l=3, weights=None):
		if not weights:
			weights = 2 * np.random.random((inputs_l, 1)) - 1

		self.weights = weights

	def calc(self, inputs):
		return sigmoid(np.dot(inputs, self.weights))

	def train(self, training_data, training_outputs, steps=10000):
		for i in range(steps):
			input_layer = training_data

			outputs = self.calc(input_layer)
			err = training_outputs - outputs

			delta = np.dot(input_layer.T, err*(outputs*(1-outputs)))
			self.weights += delta


def example1(l):
	training_data = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1]])
	training_outputs = np.array([[0,1,1,0]]).T

	percp = Perceptron()
	percp.train(training_data, training_outputs, l)

	return percp.calc([1,1,0])

def example2(l):
	training_data = np.array([[0,0,0], [0,0,1], [0,1,0]])
	training_outputs = np.array([[0,1,1]]).T

	percp = Perceptron()
	percp.train(training_data, training_outputs, l)

	return percp.calc([1,1,0])


print(example1(10000))
print(example1(30000))

print(example2(10000))
print(example2(30000))

