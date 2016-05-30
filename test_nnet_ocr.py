from input_font import glyphs
from input_image import image
from nnet_model import ocr
from keras_model import font_data, image_data


def train(model, inputs, labels, chars, epochs, e_output=100):
    for i in range(0, epochs, e_output):
        # train for 5 epochs
        t_error = model.train(inputs, labels, e_output)

        # predict class for all training data
        p_classes = model.predict(inputs)

        # get wrongly predicted chars
        p_chars = font_data.deconvert(chars, p_classes)
        p_errors = [c for c, pc in zip(chars, p_chars) if c != pc]

        print()
        print('epoch: %i' % (i + e_output))
        print('training error: %f' % t_error)
        print('predict: %s' % ''.join(p_chars))
        print('errors: %s' % ''.join(p_errors))

        #if len(p_errors) == 0:
        #    break


def test_nnet_ocr():
    g = list(glyphs.default_glyphs())
    inputs, labels, chars = font_data.convert(g)

    imgs = list(image.read('input_image/test_image_w.png', 9, 18))
    test_inputs = image_data.convert(imgs)

    model = ocr.Model(len(inputs[0]), len(labels[0]))

    # train for 100 epochs
    train(model, inputs, labels, chars, 1000)

    # predict class for test_inputs
    p_classes = model.predict(test_inputs)

    p_chars = font_data.deconvert(chars, p_classes)
    for i in range(0, len(p_chars), 80):
        print(''.join(p_chars[i:i + 80]))


if __name__ == "__main__":
    test_nnet_ocr()
