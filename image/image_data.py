import numpy as np


# convert PIL images to numpy array
def convert(images):

    num_datasets = len(images)
    num_pixels = images[0].width * images[0].height

    data = np.zeros((num_datasets, num_pixels))

    for index, sample in enumerate(images):
        data[index, :] = np.asarray(sample, dtype='float32').flatten() / 255

    return data