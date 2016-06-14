import numpy as np

# single neuron:
#
# i = num inputs
# x_i = inputs = outputs of last layer
# w_i = weights
# z = sum_i(w * x_i)
# y = activation(z) = output of neuron


class Layer(object):
    # self.w weight matrix
    def __init__(self, num_neurons, num_inputs):
        self.weight_factor = 1 / np.sqrt(num_inputs)
        # initialize random weight matrix
        self.w = (np.random.random((num_neurons, num_inputs)).astype('float64')
                  - 0.5) * self.weight_factor

    # self.x input vector
    # self.y output vector
    def feed_forward(self, input_):
        self.x = input_
        # multiply weights with output of previous layer
        self.y = np.dot(self.w, self.x)
        return self.y

    def back_propagate(self, y_error):
        # derivative of output error derivative vector multiplied by weights
        x_error = np.dot(y_error, self.w)
        # weight error = error derivative with regard to weight matrix, for correction of weight matix
        w_corr = np.multiply(np.matrix(y_error).T, self.x)
        return x_error, w_corr

    def correct_weights(self, w_corr, learning_rate):
        # correct weights by weight error and learning rate
        self.w -= np.multiply(learning_rate * self.weight_factor, w_corr)


class BiasLayer(Layer):
    def __init__(self, num_neurons, num_inputs):
        # init with additional bias input
        super().__init__(num_neurons, num_inputs + 1)

    def feed_forward(self, input_):
        # append bias to input vector
        bias_input = np.append(input_, 1)
        return super().feed_forward(bias_input)

    def back_propagate(self, y_error):
        x_error, weight_error = super().back_propagate(y_error)
        # remove bias error from x_error vector
        return x_error[:-1], weight_error


class SigmoidActivation(object):
    def feed_forward(self, input_):
        self.y = logistic(input_)
        return self.y

    def back_propagate(self, y_error):
        return np.multiply(logistic_deriv(self.y), y_error)


class SoftmaxActivation(object):
    def feed_forward(self, input_):
        self.y = softmax(input_)
        return self.y

    def back_propagate(self, y_error):
        # when using ce_softmax_error error function this step is unnecessary
        return y_error


# squared error
def sq_error(t, y):
    return y - t


# logistic activation
def logistic(z):
    return 1 / (1 + np.exp(-z))


# logistic activation derivative
def logistic_deriv(y):
    return np.multiply(y, 1 - y)


# softmax derivative of cross entropy error
def ce_softmax_error(t, y):
    # cross entropy error: -np.sum(np.multiply(y, np.log(y)))
    return y - t


# softmax activation
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))