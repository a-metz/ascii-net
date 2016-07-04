from PIL import Image, ImageDraw, ImageFont


# render a glyph to a PIL image
def generate_glyph(char, font, dim):
    img = Image.new('L', dim, color=255)
    img.char = char
    # get draw handle
    dh = ImageDraw.Draw(img)
    # draw glyph
    dh.text((0, 0), char, font=font, fill=0)
    return img


# render all glyphs in charset
def glyphs(charset, fontname, fontsize, dim):
    font = ImageFont.truetype(fontname, fontsize)
    for char in charset:
        yield generate_glyph(char, font, dim)


def default_glyphs():
    return glyphs(
        charset=
        '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;?@[\\]^_`{|}~ ',
        fontname='DejaVuSansMono.ttf',
        fontsize=15,
        dim=(9, 18))
