import image


def test_image():
    imgs = list(image.read('test_image_w.png', 8, 18))

    # test image has dimension 720x630
    assert (len(imgs) == (720 / 8) * (630 / 18))