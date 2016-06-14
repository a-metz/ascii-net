import pytest

from font import glyphs, font_data
from input.image import tiles, image_data


def train(model, inputs, labels, chars, epochs, e_output=100):
    for i in range(0, epochs, e_output):
        model.train(inputs, labels, e_output)

        # predict class for all training data
        p_classes = model.predict(inputs)

        # get wrongly predicted chars
        p_chars = font_data.deconvert(chars, p_classes)
        p_errors = [c for c, pc in zip(chars, p_chars) if c != pc]

        print()
        print('epoch: %i' % (i + e_output))
        print('predict: %s' % ''.join(p_chars))
        print('errors: %s' % ''.join(p_errors))

        #if len(p_errors) == 0:
        #    break


def generate(backend, epochs):
    if backend == 'keras':
        from model.keras import OcrModel
    elif backend == 'nnet':
        from model.nnet import OcrModel
    else:
        raise NotImplementedError

    g = list(glyphs.default_glyphs())
    inputs, labels, chars = font_data.convert(g)

    t = list(tiles.read('test/test_image_w.png', 9, 18))
    test_inputs = image_data.convert(t)

    model = OcrModel(len(inputs[0]), len(labels[0]))

    # train for x epochs
    train(model, inputs, labels, chars, epochs)

    # predict class for test_inputs
    p_classes = model.predict(test_inputs)

    p_chars = font_data.deconvert(chars, p_classes)
    for i in range(0, len(p_chars), 80):
        print(''.join(p_chars[i:i + 80]))


def test_generate_nnet():
    generate('nnet', 1)


def test_generate_keras():
    generate('keras', 1)


if __name__ == "__main__":
    generate('nnet', 2000)
    generate('keras', 2000)
