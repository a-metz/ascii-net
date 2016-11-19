import numpy as np
from . import glyphs
from . import font_data


def test_glyphs():
    charset = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;?@[\\]^_`{|}~ '
    img_glyphs = list(glyphs.glyphs(charset=charset,
                                    fontname='fonts/DejaVuSansMono.ttf',
                                    fontsize=18,
                                    dim=(11, 23)))
    assert len(img_glyphs) == len(charset)
    assert img_glyphs[0].height == 23
    assert img_glyphs[0].width == 11

    data, labels, chars = font_data.convert(img_glyphs)
    assert np.size(data, axis=0) == len(charset)
    assert np.size(data, axis=1) == 23 * 11

    print(np.argmax(labels, axis=1))
    char_list = font_data.deconvert(chars=chars,
                                    classes=np.argmax(labels, axis=1))
    assert charset == ''.join(char_list)