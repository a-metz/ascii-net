import pytest
import numpy as np

from font import training_data, param, transform, render
from image import tiles, image_data

# temporary batch loading of training data from new training data generator
def default_training_data():
    fnt = render.load_font(fontname=param.DEFAULT_FONT, fontsize=param.DEFAULT_SIZE)
    trn = transform.random_affine(param.DEFAULT_TRANSPOSE_SCALE, param.DEFAULT_AFFINE_SCALE)
    gen, get_char, get_index = training_data.get_sample_generator(charset=param.PRINTABLE_ASCII, font=fnt, dim=param.DEFAULT_DIM,
            offset=param.DEFAULT_OFFSET, transform=trn)

    batch_size = 92
    batch = list(training_data.batch(gen, batch_size))
    input_len = len(batch[0][1].flatten())
    inputs = np.zeros((batch_size, input_len))
    label_len = len(batch[0][0])
    labels = np.zeros((batch_size, label_len))

    for index, tple in enumerate(batch):
        inputs[index, :] = tple[1].flatten()
        labels[index, :] = tple[0]
    
    chars = [get_char(tple[0]) for tple in batch]

    return inputs, labels, chars, get_char, get_index


def train(model, inputs, labels, epochs, e_output=100):
    for i in range(0, epochs, e_output):
        train_loss = model.train(inputs, labels, e_output)

        # predict class for all training data
        p_classes = model.predict(inputs)

        print('epoch: %i, loss: %f' % (i + e_output, train_loss))
        #print('predict: %s' % ''.join(p_chars))
        #print('errors: %s' % ''.join(p_errors))
        #if len(p_errors) == 0:
        #    break


def generate(backend, epochs):
    if backend == 'keras':
        from model.keras import OcrModel
    elif backend == 'nnet':
        from model.nnet import OcrModel
    else:
        raise NotImplementedError

    print('generate with %s model in %d epochs' % (backend, epochs))

    print('load data')
    inputs, labels, chars, get_char, get_index = default_training_data()

    t = list(tiles.read('test/test_image_w_inv.png', 9, 18))
    test_inputs = image_data.convert(t)

    print('load model')
    model = OcrModel(len(inputs[0]), len(labels[0]))

    print('start training')
    # train for x epochs
    train(model, inputs, labels, epochs)

    print('predict')
    # predict class for test_inputs
    p_classes = model.predict(test_inputs)

    p_chars = [get_index(cl) for cl in p_classes]
    for i in range(0, len(p_chars), 80):
        print(''.join(p_chars[i:i + 80]))


def test_generate_nnet():
    generate('nnet', 100)


def test_generate_keras():
    generate('keras', 100)


if __name__ == "__main__":
    #generate('nnet', 1000)
    generate('keras', 5000)
