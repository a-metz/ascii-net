import image


def test_image():
    imgs = list(image.read('test_image_w.png', 11, 23))

    # test image has dimension 440x460
    assert (len(imgs) == 40 * 20)