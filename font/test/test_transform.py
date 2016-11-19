import glyphs

charset = 'ABC'
glyphs = list(glyphs.glyphs(
    charset=charset,
    fontname='fonts/DejaVuSansMono.ttf',
    fontsize=18,
    dim=(11, 23)))


affine_matrix = [1, 0, 10,
                 0, 1, 0]

glyphs[0].transform(size, PIL.Image.AFFINE, data=affine_matrix, resample=1, fill=0)

