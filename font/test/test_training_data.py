import font
from font import param
from font import render
from font import training_data
from font import transform
from tools import console


def test():
    fnt = render.load_font(fontname=param.DEFAULT_FONT, fontsize=param.DEFAULT_SIZE)
    trns = transform.random_affine(param.DEFAULT_TRANSPOSE_SCALE, param.DEFAULT_AFFINE_SCALE)
    gen, _, _ = training_data.get_sample_generator(charset=param.PRINTABLE_ASCII, font=fnt, dim=param.DEFAULT_DIM,
            offset=param.DEFAULT_OFFSET, transform=trns)

    assert len(list(training_data.batch(gen, 100))) == 100

    for lbl, gly in training_data.batch(gen, 100):
        console.draw(gly)
        print()


if __name__ == '__main__':
    test()