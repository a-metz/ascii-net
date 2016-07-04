import numpy as np


# convert glyphs to numpy arrays of image data, one-hot class labels and
# corresponding characters
def convert(glyphs):
    # infer dimensions from glyphs
    num_pixels = glyphs[0].width * glyphs[0].height
    num_classes = len(glyphs)
    num_datasets = num_classes

    # initialize arrays
    data = np.zeros((num_datasets, num_pixels))
    labels = np.zeros((num_datasets, num_classes))
    chars = []

    for index, glyph in enumerate(glyphs):
        # convert to flat numpy array
        data[index, :] = np.asarray(glyph, dtype='float32').flatten() / 255
        # generate one-hot class label
        label = np.zeros(num_classes)
        label[index] = 1
        labels[index, :] = label

        chars.append(glyph.char)

    return data, labels, chars


# deconvert array of classes to characters
def deconvert(chars, classes):
    return [chars[c] for c in classes]