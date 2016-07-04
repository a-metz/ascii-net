import numpy as np
from . import tiles
from . import image_data


def test_read():
    img_tiles = list(tiles.read('test/test_image_w.png', 8, 18))

    # test image has dimension 720x630
    # check number of image tiles
    assert (len(img_tiles) == (720 / 8) * (630 / 18))

    data = image_data.convert(img_tiles)

    # check size of data numpy array
    assert (np.size(data, axis=0) == (720 / 8) * (630 / 18))
    assert (np.size(data, axis=1) == 8 * 18)