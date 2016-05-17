import numpy as np
from PIL import Image


def read(filename, sample_width, sample_height):
    x_off = 0
    y_off = 0
    im = Image.open(filename).convert('L')

    for y_off in range(0, im.height - sample_height, sample_height):
        for x_off in range(0, im.width - sample_width, sample_width):

            # extract sample image from whole image
            sample_im = im.crop((x_off, y_off, x_off + sample_width, y_off +
                                 sample_height))
            yield sample_im
