import itertools
import numpy as np
from . import render
from . import param
from . import transform


# training sample generator
def get_sample_generator(charset, font, dim, offset, transform=lambda x: x, loop=True):
    # render all chars in charset and generate labels
    samples = []
    chars = {}
    for index, char in enumerate(charset):
        # numpy array of rendered glyph
        glyph = render.convert_to_array(render.render_glyph_image(char, font, dim, offset))
        # class label as one-got vector
        label = np.zeros(len(charset))[index] = 1
        # add dataset to sample list
        samples.append((label, glyph))
        # add label to char mapping
        chars[label] = char

    # generate glyph for training
    def gen():
        for label, glyph in itertools.cycle(samples) if loop else samples:
            yield label, transform(glyph)

    # get char from label
    def char(label):
        return char[label]

    return gen, char


def batch_generator(gen, batch_size):
    for _, out in zip(range(batch_size), gen()):
        yield out
