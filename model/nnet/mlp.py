import numpy as np

from nnet_model import nnet


class ClassicMLP(object):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        # initialize layers and activations
        self.layer_hidden = nnet.BiasLayer(num_neurons=num_hidden,
                                           num_inputs=num_inputs)
        self.activ_hidden = nnet.SigmoidActivation()

        self.layer_output = nnet.BiasLayer(num_neurons=num_outputs,
                                           num_inputs=num_hidden)
        self.activ_output = nnet.SigmoidActivation()

        # set error function
        self.error_func = nnet.sq_error

    def evaluate(self, input_):
        z_hidden = self.layer_hidden.feed_forward(input_)
        y_hidden = self.activ_hidden.feed_forward(z_hidden)
        z_output = self.layer_output.feed_forward(y_hidden)
        y_output = self.activ_output.feed_forward(z_output)
        return y_output

    def get_weight_errors(self, input_, expected_output):
        y_error_output = self.error_func(expected_output,
                                         self.evaluate(input_))

        # back propagate output layer
        z_error_output = self.activ_output.back_propagate(y_error_output)
        x_error_output, wcorr_output = self.layer_output.back_propagate(
            z_error_output)

        # back propagate hidden layer
        z_error_hidden = self.activ_hidden.back_propagate(x_error_output)
        x_error_hidden, wcorr_hidden = self.layer_hidden.back_propagate(
            z_error_hidden)

        return (wcorr_output, wcorr_hidden)

    def train_online(self, input_, expected_output, learning_rate):
        # get weight corrections by evaluation and back propagation
        wcorr_output, wcorr_hidden = self.get_weight_errors(input_,
                                                            expected_output)

        self.layer_output.correct_weights(acc_wcorr_output,
                                          learning_rate=learning_rate)
        self.layer_hidden.correct_weights(acc_wcorr_hidden,
                                          learning_rate=learning_rate)

        # return training error
        return np.mean(np.abs(expected_output - self.evaluate(input_)))

    def train_batch(self, inputs, expected_outputs, learning_rate):
        acc_wcorr_output = np.zeros_like(self.layer_output.w)
        acc_wcorr_hidden = np.zeros_like(self.layer_hidden.w)

        for input_, expected_output in zip(inputs, expected_outputs):
            # get weight errors by evaluation and back propagation
            wcorr_output, wcorr_hidden = self.get_weight_errors(
                input_, expected_output)

            # accumulate weight errors
            acc_wcorr_output += wcorr_output
            acc_wcorr_hidden += wcorr_hidden

        self.layer_output.correct_weights(acc_wcorr_output,
                                          learning_rate=learning_rate)
        self.layer_hidden.correct_weights(acc_wcorr_hidden,
                                          learning_rate=learning_rate)

        return np.mean(np.abs(expected_output - self.evaluate(input_)))


class SoftmaxMLP(ClassicMLP):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        # initialize layers and activations
        self.layer_hidden = nnet.BiasLayer(num_neurons=num_hidden,
                                           num_inputs=num_inputs)
        self.activ_hidden = nnet.SigmoidActivation()

        self.layer_output = nnet.BiasLayer(num_neurons=num_outputs,
                                           num_inputs=num_hidden)
        self.activ_output = nnet.SoftmaxActivation()

        # set error function
        self.error_func = nnet.ce_softmax_error