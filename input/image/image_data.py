import numpy as np


def convert(tiles):
    num_datasets = len(tiles)
    num_pixels = tiles[0].width * tiles[0].height

    input_tiles = np.zeros((num_datasets, num_pixels))

    for index, sample in enumerate(tiles):
        input_tiles[index, :] = np.asarray(sample,
                                           dtype='float32').flatten() / 255

    return input_tiles