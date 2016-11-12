from PIL import Image, ImageDraw, ImageFont
import numpy as np

 # render a glyph to a PIL image
def render_glyph_image(char, font, dim):
    img = Image.new('L', dim, color=0)
    # get draw handle
    dh = ImageDraw.Draw(img)
    # draw glyph
    dh.text((0, 0), char, font=font, fill=255)
    return img


def convert_to_array(image):
    return np.asarray(image, dtype='float32') / 255


# render all glyphs in charset
def generate_glyphs(charset, fontname, fontsize, dim):
    font = ImageFont.truetype(fontname, fontsize)
    for char in charset:
        glyph = convert_to_array(render_glyph_image(char, font, dim))
        yield char, glyph
