import numpy as np


def convert(images):
    num_datasets = len(images)
    num_pixels = images[0].width * images[0].height

    test_inputs = np.zeros((num_datasets, num_pixels))

    for index, sample in enumerate(images):
        test_inputs[index, :] = np.asarray(sample,
                                           dtype='float32').flatten() / 255

    return test_inputs