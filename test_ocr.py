import pytest
import numpy as np

from font import training_data, param, transform, render
from image import tiles, image_data


# generate one batch of training data
def generate_training_data(batch_size):
    fnt = render.load_font(fontname=param.DEFAULT_FONT, fontsize=param.DEFAULT_SIZE)
    
    # disable transform until it actually build a model where it helps the results
    # trn = transform.random_affine(param.DEFAULT_TRANSPOSE_SCALE, param.DEFAULT_AFFINE_SCALE)
    trn = lambda x: x

    gen, get_char, get_index = training_data.get_sample_generator(charset=param.PRINTABLE_ASCII, font=fnt, dim=param.DEFAULT_DIM,
            offset=param.DEFAULT_OFFSET, transform=trn)

    batch = list(training_data.batch(gen, batch_size))
    inputs_dim = (batch_size, ) + batch[0][1].shape
    inputs = np.zeros(inputs_dim)
    labels_dim = (batch_size, len(batch[0][0]))
    labels = np.zeros(labels_dim)

    for index, tple in enumerate(batch):
        inputs[index, :, :] = tple[1]
        labels[index, :] = tple[0]
    
    chars = [get_char(tple[0]) for tple in batch]

    return inputs, labels, chars, get_char, get_index


def generate(backend, batch_size, epochs):
    if backend == 'keras_mlp':
        from model.keras_mlp import OcrModel
    elif backend == 'keras_cnn':
        from model.keras_cnn import OcrModel
    elif backend == 'nnet':
        from model.nnet import OcrModel
    else:
        raise NotImplementedError

    print('generate with %s model in %d epochs' % (backend, epochs))

    t = list(tiles.read('test/test_image_w_inv.png', 9, 18))
    test_inputs = image_data.convert(t)
    # shift mean
    test_inputs = test_inputs - 0.4

    print('load model')
    inputs, labels, _, _, _ = generate_training_data(1)
    model = OcrModel(inputs[0].shape, len(labels[0]))

    print('start training')
    
    for i in range(0, epochs):
        # generate batch of training data
        inputs, labels, _, _, get_index = generate_training_data(batch_size)
        
        # shift mean
        inputs = inputs - 0.4
        # train
        train_loss = model.train(inputs, labels)

        # predict class for all training data
        p_classes = model.predict(inputs)

        print('epoch: %i, loss: %f' % (i, train_loss))

        print('predict')
        # predict class for test_inputs
        p_classes = model.predict(test_inputs)

        p_chars = [get_index(cl) for cl in p_classes]
        for i in range(0, len(p_chars), 80):
            print(''.join(p_chars[i:i + 80]))


CHARSET_SIZE = len(param.PRINTABLE_ASCII)


def test_generate_nnet():
    generate('nnet', batch_size=CHARSET_SIZE, epochs=1)


def test_generate_keras_mlp():
    generate('keras_mlp', batch_size=CHARSET_SIZE, epochs=1)


def test_generate_keras_cnn():
    generate('keras_cnn', batch_size=CHARSET_SIZE, epochs=1)


if __name__ == "__main__":
    generate('keras_mlp', batch_size=CHARSET_SIZE, epochs=200)
