import numpy as np

# single neuron:
#
# x_i = inputs = outputs of last layer
# w_i = weights
# z = sum_i(x_i * w_i)
# y = nonlin(z) = output of current layer


class Layer:
	# n number of neurons
	# i number of inputs
	# self.w weight matrix
	def __init__(self, n, i):
		# initialize random weight matrix
		self.w = (2 * np.random.random((n, i))) - 1

	# self.x input vector
	# self.y output vector
	def feed_forward(self, x):
		self.x = x
		# multiply weights with output of previous layer
		z = np.dot(self.w, self.x)
		# transform with nonlinar function
		self.y = nonlin(z)

		return self.y

	# self.dE_y error derivative vector
	# l learning rate
	def back_propagate(self, dE_y, l=1):
		# error derivative vector with regard to sum of weighted inputs z
		dE_z = np.multiply(nonlin_deriv(self.y), dE_y)
		# error derivate vector multiplied by weights for previous layer
		dE_x = np.dot(self.w.T, dE_z)
		# error derivative with regard to weight matrix, for correction of weight matix
		dE_w = np.multiply(self.x, np.matrix(dE_z).T)
		# correct weights by learning rate
		self.w += np.multiply(l, dE_w)

		return dE_x


# nonlinear function, y(z)
def nonlin(z):
	return 1 / (1 + np.exp(-z))

# derivative of nonlinear function, dy/dz
def nonlin_deriv(y):
	return y * (1 - y)
