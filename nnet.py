import numpy as np

# single neuron:
#
# x_i = inputs = outputs of last layer
# w_i = weights
# z = sum_i(x_i * w_i)
# y = nonlin(z) = output of current layer


class Layer(object):
    # n number of neurons
    # i number of inputs
    # self.w weight matrix
    def __init__(self, num_neurons, num_inputs, weight_transform=lambda w: w):
        self.weight_factor = 1 / np.sqrt(num_inputs)
        # initialize random weight matrix
        self.w = (np.random.random((num_neurons, num_inputs)) -
                  0.5) * self.weight_factor
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

    def back_propagate(self, output_error):
        # logit error = error derivative vector with regard to sum of weighted inputs z
        delta = np.multiply(nonlin_deriv(self.y), output_error)
        # transform weights
        w = self.numpy_weight_transform(self.w)
        # input error derivate vector multiplied by weights
        input_error = np.dot(delta, w)
        # weight error = error derivative with regard to weight matrix, for correction of weight matix
        weight_error = np.multiply(np.matrix(delta).T, self.x)
        return (input_error, weight_error)

    def correct_weights(self, weight_error, learning_rate):
        # correct weights by weight error and learning rate
        self.w += np.multiply(learning_rate * self.weight_factor, weight_error)


# nonlinear function, y(z)
def nonlin(z):
    return 1 / (1 + np.exp(-z))


# derivative of nonlinear function, dy/dz
def nonlin_deriv(y):
    return np.multiply(y, 1 - y)
