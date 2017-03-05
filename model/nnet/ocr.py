import numpy as np

from .mlp import SoftmaxMLP


class OcrModel(object):
    def __init__(self, shape_pixels, num_classes):
        # flattend input shape
        self.num_pixels = shape_pixels[0] * shape_pixels[1]

        self.model = SoftmaxMLP(num_inputs=self.num_pixels,
                                num_hidden=self.num_pixels,
                                num_outputs=num_classes)

    def flatten_pixels(self, inputs):
        return inputs.reshape((-1, self.num_pixels))

    def train(self, inputs, labels, epochs=1):
        for i in range(epochs):
            train_error = self.model.train_batch(
                self.flatten_pixels(inputs), labels, learning_rate=0.5)
        return train_error

    def predict(self, inputs):

        res = np.argmax(
            np.apply_along_axis(
                self.model.evaluate,
                axis=1,
                arr=self.flatten_pixels(inputs)),
            axis=1)

        return res