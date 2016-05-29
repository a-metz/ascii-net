import numpy as np

import nnet
from classic_mlp_model import ClassicMLPModel


class SoftmaxMLPModel(ClassicMLPModel):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        # initialize layers and activations
        self.layer_hidden = nnet.BiasLayer(num_neurons=num_hidden,
                                           num_inputs=num_inputs)
        self.activ_hidden = nnet.SigmoidActivation()

        self.layer_output = nnet.BiasLayer(num_neurons=num_outputs,
                                           num_inputs=num_hidden)
        self.activ_output = nnet.SoftmaxActivation()

        self.error_func = nnet.ce_softmax_error