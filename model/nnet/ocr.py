import numpy as np

from .mlp import SoftmaxMLP


class OcrModel(object):
    def __init__(self, num_pixels, num_classes):
        self.model = SoftmaxMLP(num_inputs=num_pixels,
                                num_hidden=num_pixels,
                                num_outputs=num_classes)

    def train(self, inputs, labels, epochs=1):
        for i in range(epochs):
            train_error = self.model.train_batch(
                inputs, labels, learning_rate=0.5)
        return train_error

    def predict(self, inputs):

        res = np.argmax(
            np.apply_along_axis(self.model.evaluate,
                                axis=1, arr=inputs),
            axis=1)

        return res