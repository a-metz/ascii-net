import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

batch_size = 128

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

class OcrModel(object):
    def __init__(self, shape_pixels, num_classes):
        self.input_shape = shape_pixels + (1, )

        self.model = Sequential()
        self.model.add(Convolution2D(nb_filters,
                                     kernel_size[0],
                                     kernel_size[1],
                                     border_mode='valid',
                                     input_shape=self.input_shape))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=pool_size))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adadelta')

    def reshape_inputs(self, inputs):
        return inputs.reshape((-1, ) + self.input_shape)

    def train(self, inputs, labels, epochs=1):
        history = self.model.fit(self.reshape_inputs(inputs),
                                 labels,
                                 batch_size=batch_size,
                                 nb_epoch=epochs,
                                 verbose=1)
        return history.history['loss'][-1]

    def predict(self, inputs):
        return self.model.predict_classes(self.reshape_inputs(inputs), verbose=0)
