import scipy.ndimage as ndimage
import numpy as np


def random_affine(glyph, transpose_max, affine_max):
    # random transpose: +-transpose_max
    transpose = (np.random.random(2) - 0.5) * 2 * transpose_max
    
    # identity matrix (no transform)
    affine = np.asarray([[1, 0], [0, 1]], dtype='float32')
    # add randomness: +-affine_max
    affine += (np.random.random((2,2)) - 0.5) * 2 * affine_max
    
    return ndimage.affine_transform(glyph, matrix=affine, offset=transpose, cval=0)