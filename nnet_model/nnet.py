import numpy as np

# single neuron:
#
# x_i = inputs = outputs of last layer
# w_i = weights
# z = sum_i(x_i * w_i)
# y = nonlin(z) = output of current layer


class Layer(object):
    # self.w weight vector
    def __init__(self, num_neurons, num_inputs, weight_transform=lambda w: w):
        self.weight_factor = 1 / np.sqrt(num_inputs)
        # initialize random weight matrix
        self.w = (np.random.random((num_neurons, num_inputs)) -
                  0.5) * self.weight_factor
        self.numpy_weight_transform = np.vectorize(weight_transform)

    # self.x input vector
    # self.y output vector
    def feed_forward(self, input_, activation):
        self.x = input_
        # transform weights
        w = self.numpy_weight_transform(self.w)
        # multiply weights with output of previous layer
        z = np.dot(w, self.x)
        # transform with nonlinar function
        self.y = activation(z)
        return self.y

    def back_propagate(self, output_error, activation_deriv):
        # logit error = error derivative vector with regard to sum of weighted inputs z
        delta = np.multiply(activation_deriv(self.y), output_error)
        # transform weights
        w = self.numpy_weight_transform(self.w)
        # input error derivative vector multiplied by weights
        input_error = np.dot(delta, w)
        # weight error = error derivative with regard to weight matrix, for correction of weight matix
        weight_error = np.multiply(np.matrix(delta).T, self.x)
        return (input_error, weight_error)

    def correct_weights(self, weight_error, learning_rate):
        # correct weights by weight error and learning rate
        self.w -= np.multiply(learning_rate * self.weight_factor, weight_error)


# squared error cost gradient
def error(t, y):
    return y - t


# logistic activation
def logistic(z):
    return 1 / (1 + np.exp(-z))


# logistic activation derivative
def logistic_deriv(y):
    return np.multiply(y, 1 - y)


# softmax activation
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


# cross entropy softmax derivative
def ce_softmax_deriv(y):
    return 1
