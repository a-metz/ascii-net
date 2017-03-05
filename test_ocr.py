import pytest
import numpy as np

from font import training_data, param, transform, render
from image import tiles, image_data


# generate one batch of training data
def generate_training_data(batch_size):
    fnt = render.load_font(fontname=param.DEFAULT_FONT, fontsize=param.DEFAULT_SIZE)
    #trn = transform.random_affine(param.DEFAULT_TRANSPOSE_SCALE, param.DEFAULT_AFFINE_SCALE)
    #gen, get_char, get_index = training_data.get_sample_generator(charset=param.PRINTABLE_ASCII, font=fnt, dim=param.DEFAULT_DIM,
    #        offset=param.DEFAULT_OFFSET, transform=trn)

    gen, get_char, get_index = training_data.get_sample_generator(charset=param.PRINTABLE_ASCII, font=fnt, dim=param.DEFAULT_DIM,
            offset=param.DEFAULT_OFFSET)

    batch = list(training_data.batch(gen, batch_size))
    inputs_len = len(batch[0][1].flatten())
    inputs = np.zeros((batch_size, inputs_len))
    labels_len = len(batch[0][0])
    labels = np.zeros((batch_size, labels_len))

    for index, tple in enumerate(batch):
        inputs[index, :] = tple[1].flatten()
        labels[index, :] = tple[0]
    
    chars = [get_char(tple[0]) for tple in batch]

    return inputs, labels, chars, get_char, get_index


def generate(backend, batch_size, epochs):
    if backend == 'keras_mlp':
        from model.keras_mlp import OcrModel
    elif backend == 'nnet':
        from model.nnet import OcrModel
    else:
        raise NotImplementedError

    print('generate with %s model in %d epochs' % (backend, epochs))

    t = list(tiles.read('test/test_image_w_inv.png', 9, 18))
    test_inputs = image_data.convert(t)

    print('load model')
    inputs, labels, _, _, _ = generate_training_data(1)
    model = OcrModel(len(inputs[0]), len(labels[0]))

    print('start training')
    
    for i in range(0, epochs):
        # generate batch of training data
        inputs, labels, _, _, get_index = generate_training_data(batch_size)
        
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


def test_generate_nnet():
    generate('nnet', 100, 1)


def test_generate_keras_mlp():
    generate('keras_mlp', 100, 1)


if __name__ == "__main__":
    #generate('nnet', 1000)
    charset_size = len(param.PRINTABLE_ASCII)
    generate('keras_mlp', batch_size=charset_size * 100, epochs=10000)
