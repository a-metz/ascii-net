import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD


def test_random_binary():
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])

    labels = np.asarray([[0], [1], [1], [0]])

    model = Sequential()
    model.add(Dense(output_dim=2, input_dim=2))
    model.add(Activation('sigmoid'))
    model.add(Dense(output_dim=1))
    model.add(Activation('sigmoid'))

    model.compile(loss='mean_squared_error', optimizer=SGD(lr=10))

    model.fit(inputs, labels, batch_size=1, nb_epoch=1000)

    print(model.evaluate(inputs, labels, batch_size=1))
    print(model.predict(inputs, batch_size=1))


if __name__ == "__main__":
    test_random_binary()