import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD


class OcrModel(object):
    def __init__(self, shape_pixels, num_classes):
        # flattend input shape
        self.num_pixels = shape_pixels[0] * shape_pixels[1]

        self.model = Sequential()
        self.model.add(Dense(output_dim=self.num_pixels, input_dim=self.num_pixels))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dense(output_dim=num_classes))
        self.model.add(Activation('softmax'))

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(lr=1, momentum=0.9, nesterov=True))

    def flatten_pixels(self, inputs):
        return inputs.reshape((-1, self.num_pixels))

    def train(self, inputs, labels, epochs=1):
        history = self.model.fit(self.flatten_pixels(inputs),
                                 labels,
                                 batch_size=100,
                                 nb_epoch=epochs,
                                 verbose=0)
        # return loss of last epoch
        return history.history['loss'][-1]

    def predict(self, inputs):
        return self.model.predict_classes(self.flatten_pixels(inputs), verbose=0)
