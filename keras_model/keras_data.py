import numpy as np


def convert(glyphs):
    # infer dimensions from glyphs
    num_pixels = glyphs[0].width * glyphs[0].height
    num_classes = len(glyphs)
    num_datasets = num_classes

    # initialize arrays
    inputs = np.zeros((num_datasets, num_pixels))
    labels = np.zeros((num_datasets, num_classes))
    chars = []

    for index, glyph in enumerate(glyphs):
        # convert to flat numpy array
        inputs[index, :] = np.asarray(glyph, dtype='float32').flatten() / 255
        # generate multinomial class label
        label = np.zeros(num_classes)
        label[index] = 1
        labels[index, :] = label

        chars.append(glyph.char)

    return inputs, labels, chars


def deconvert(chars, classes):
    return [chars[c] for c in classes]