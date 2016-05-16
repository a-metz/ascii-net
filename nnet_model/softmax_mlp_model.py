import numpy as np

import nnet


class SoftmaxMLPModel(object):
    def __init__(self, num_inputs, num_hidden, num_outputs):

        # initialize layers with bias inputs
        self.hidden_layer = nnet.Layer(num_neurons=num_hidden,
                                       num_inputs=num_inputs + 1)

        self.output_layer = nnet.Layer(num_neurons=num_outputs,
                                       num_inputs=num_hidden + 1)

        self.softmax_layer = nnet.Layer(num_neurons=num_outputs,
                                        num_inputs=num_outputs)

    def evaluate(self, input_):
        # append bias to input vector
        input_ = np.append(input_, 1)
        hidden = self.hidden_layer.feed_forward(input_,
                                                activation=nnet.logistic)
        # append bias to hidden vector
        hidden = np.append(hidden, 1)
        output = self.output_layer.feed_forward(hidden,
                                                activation=nnet.logistic)

        softmax = self.softmax_layer.feed_forward(output,
                                                  activation=nnet.softmax)
        return softmax

    def get_weight_errors(self, input_, expected_output):
        softmax_error = nnet.error(expected_output, self.evaluate(input_))
        # backpropagate softmax layer
        output_error, softmax_weight_error = self.softmax_layer.back_propagate(
            softmax_error,
            activation_deriv=nnet.ce_softmax_deriv)
        # backpropagate output layer
        hidden_error, output_weight_error = self.output_layer.back_propagate(
            output_error, activation_deriv=nnet.logistic_deriv)
        # remove bias error from hidden_error vector
        hidden_error = hidden_error[:-1]
        # backpropagate hidden layer
        input_error, hidden_weight_error = self.hidden_layer.back_propagate(
            hidden_error, activation_deriv=nnet.logistic_deriv)
        # remove bias error from input_error vector
        #input_error = input_error[:-1]

        return (softmax_weight_error, output_weight_error, hidden_weight_error)

    # def train_online(self, input_, expected_output, learning_rate):
    #     # get weight errors by evaluation and back propagation
    #     output_weight_error, hidden_weight_error = self.get_weight_errors(
    #         input_, expected_output)

    #     self.output_layer.correct_weights(acc_output_weight_error,
    #                                       learning_rate=learning_rate)
    #     self.hidden_layer.correct_weights(acc_hidden_weight_error,
    #                                       learning_rate=learning_rate)

    #     # return training error
    #     return np.mean(np.abs(expected_output - self.evaluate(input_)))

    def train_batch(self, inputs, expected_outputs, learning_rate):
        acc_softmax_weight_error = np.zeros_like(self.softmax_layer.w)
        acc_output_weight_error = np.zeros_like(self.output_layer.w)
        acc_hidden_weight_error = np.zeros_like(self.hidden_layer.w)

        for input_, expected_output in zip(inputs, expected_outputs):
            # get weight errors by evaluation and back propagation
            softmax_weight_error, output_weight_error, hidden_weight_error = self.get_weight_errors(
                input_, expected_output)

            # accumulate weight errors
            acc_softmax_weight_error += softmax_weight_error
            acc_output_weight_error += output_weight_error
            acc_hidden_weight_error += hidden_weight_error

        self.softmax_layer.correct_weights(acc_softmax_weight_error,
                                           learning_rate=learning_rate)
        self.output_layer.correct_weights(acc_output_weight_error,
                                          learning_rate=learning_rate)
        self.hidden_layer.correct_weights(acc_hidden_weight_error,
                                          learning_rate=learning_rate)

        return np.mean(np.abs(expected_output - self.evaluate(input_)))