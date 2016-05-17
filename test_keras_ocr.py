from input_font import glyphs
from input_image import image
from keras_model import ocr, font_data, image_data


def test_train():
    g = list(glyphs.default_glyphs())
    inputs, labels, chars = font_data.convert(g)

    m = ocr.Model(len(inputs[0]), len(labels[0]))

    for i in range(100):
        # train for 5 epochs
        m.train(inputs, labels, 5)

        # predict class for all training data
        p_classes = m.predict(inputs)

        # get wrongly predicted chars
        p_chars = font_data.deconvert(chars, p_classes)
        p_errors = [c for c, pc in zip(chars, p_chars) if c != pc]

        print('predict:', ''.join(p_chars))
        print('errors: ', ''.join(p_errors))

        if len(p_errors) == 0:
            break


def test_generate():
    g = list(glyphs.default_glyphs())
    inputs, labels, chars = font_data.convert(g)

    imgs = list(image.read('input_image/test_image_w.png', 9, 18))
    test_inputs = image_data.convert(imgs)

    m = ocr.Model(len(inputs[0]), len(labels[0]))

    # train for 100 epochs
    m.train(inputs, labels, 1000)

    # predict class for test_inputs
    p_classes = m.predict(test_inputs)

    p_chars = font_data.deconvert(chars, p_classes)
    for i in range(0, len(p_chars), 80):
        print(''.join(p_chars[i:i + 80]))


if __name__ == "__main__":
    test_generate()
