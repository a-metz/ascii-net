import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD


class Model(object):
    def __init__(self, num_pixels, num_classes):
        self.model = Sequential()
        self.model.add(Dense(output_dim=num_pixels, input_dim=num_pixels))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dense(output_dim=num_classes))
        self.model.add(Activation('softmax'))

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(lr=1, decay=1e-6,
                          momentum=0.9,
                          nesterov=True))

    def train(self, inputs, labels, epochs):
        self.model.fit(inputs,
                       labels,
                       batch_size=92,
                       nb_epoch=epochs,
                       verbose=1)

    def predict(self, inputs):
        return self.model.predict_classes(inputs, verbose=0)
