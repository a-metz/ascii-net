import glyphs


def test_glyphs():
    g = glyphs.generate_glyphs(
        charset=
        '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;?@[\\]^_`{|}~ ',
        fontname='fonts/DejaVuSansMono.ttf',
        fontsize=18,
        dim=(11, 23))
    assert (len(g) == 92)
    assert (g[0].height == 23)
    assert (g[0].width == 11)