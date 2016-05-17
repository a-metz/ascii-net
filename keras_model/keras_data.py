import numpy as np


def convert(glyphs):

    num_pixels = glyphs[0].width * glyphs[0].height
    num_classes = len(glyphs)
    num_datasets = num_classes

    inputs = np.zeros((num_datasets, num_pixels))
    labels = np.zeros((num_datasets, num_classes))
    chars = []

    for index, glyph in enumerate(glyphs):
        inputs[index, :] = np.asarray(glyph, dtype='float32').flatten() / 255
        label = np.zeros(num_classes)
        label[index] = 1
        labels[index, :] = label
        chars.append(glyph.char)

    return inputs, labels, chars


def deconvert(chars, classes):
    return [chars[c] for c in classes]