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
	def __init__(self, num_neurons, num_inputs, weight_transform=lambda w: w):
		# initialize random weight matrix
		self.w = (2 * np.random.random((num_neurons, num_inputs))) - 1
		self.numpy_weight_transform = np.vectorize(weight_transform)

	# self.x input vector
	# self.y output vector
	def feed_forward(self, input_):
		self.x = input_
		# transform weights
		w = self.numpy_weight_transform(self.w)
		# multiply weights with output of previous layer
		z = np.dot(w, self.x)
		# transform with nonlinar function
		self.y = nonlin(z)

		return self.y

	# self.dE_y error derivative vector
	# l learning rate
	def back_propagate(self, error, learning_rate=1):
		dE_y = error
		# error derivative vector with regard to sum of weighted inputs z
		dE_z = np.multiply(nonlin_deriv(self.y), dE_y)
		# transform weights
		w = self.numpy_weight_transform(self.w)
		# error derivate vector multiplied by weights for previous layer
		dE_x = np.dot(w.T, dE_z)
		# error derivative with regard to weight matrix, for correction of weight matix
		dE_w = np.multiply(self.x, np.matrix(dE_z).T)
		# correct original weights by learning rate
		self.w += np.multiply(learning_rate, dE_w)

		return dE_x


# nonlinear function, y(z)
def nonlin(z):
	return 1 / (1 + np.exp(-z))

# derivative of nonlinear function, dy/dz
def nonlin_deriv(y):
	return y * (1 - y)
