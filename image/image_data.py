import numpy as np


# convert PIL images to numpy array
def convert(images):

    num_datasets = len(images)

    data = np.zeros((num_datasets, images[0].height, images[0].width))

    for index, sample in enumerate(images):
        data[index, :, :] = np.asarray(sample, dtype='float32') / 255 - 0.5
    return data