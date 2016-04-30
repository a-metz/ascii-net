import numpy as np

# single neuron:
#
# x_i = inputs = outputs of last layer
# w_i = weights
# z = sum_i(x_i * w_i)
# y = nonlin(z) = output of current layer

# nonlinear function, output(sum of weighted inputs)
def nonlin(z):
	return 1 / (1 + numpy.exp(-z))

# derivative of nonlinear function, dE/dw
def nonlin_deriv(y):
	return y * (1 - y)

class layer:
	# n number of neurons
	# i number of inputs
	def __init__(self, n, i):
		# initialize random weight matrix
		self.w = (2 * np.random.random((n, i))) - 1

	def feed_forward(self):
		# multiply weights with output of input_layer
		self.z = np.dot(self.w, self.x)

		# transform with nonlinar function
		self.y = nonlin(self.z)