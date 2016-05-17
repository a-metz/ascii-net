from PIL import Image, ImageDraw, ImageFont


def generate_glyph(char, font, dim):
    img = Image.new('L', dim, color=255)
    img.char = char
    dh = ImageDraw.Draw(img)
    dh.text((0, 0), char, font=font, fill=0)
    return img


def generate_glyphs(charset, fontname, fontsize, dim):
    font = ImageFont.truetype(fontname, fontsize)
    return [generate_glyph(char, font, dim) for char in charset]


def generate_default_glyphs():
    return generate_glyphs(
        charset=
        '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;?@[\\]^_`{|}~ ',
        fontname='fonts/DejaVuSansMono.ttf',
        fontsize=18,
        dim=(11, 23))
