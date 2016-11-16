import scipy.ndimage as ndimage
import numpy as np


def random_affine(transpose_scale, affine_scale, fill=0):
    def transform(glyph):
        # random transpose: +-transpose_max
        transpose = np.random.normal(loc=0, scale=transpose_scale, size=2)
        
        # identity matrix (no transform)
        affine = np.asarray([[1, 0], [0, 1]], dtype='float32')
        # add randomness: +-affine_max
        affine += np.random.normal(loc=0, scale=affine_scale, size=(2,2))

        return ndimage.affine_transform(glyph, matrix=affine, offset=transpose, cval=fill)

    return transform