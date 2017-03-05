from PIL import Image, ImageDraw, ImageFont
import numpy as np

 # render a glyph to a PIL image
def render_glyph_image(char, font, dim, offset=(0, 0)):
    # switch dimensions (PIL is wierd)
    img = Image.new('L', (dim[1], dim[0]), color=0)
    # get draw handle
    dh = ImageDraw.Draw(img)
    # draw glyph
    dh.text(offset, char, font=font, fill=255)
    return img


def convert_to_array(image):
    return np.asarray(image, dtype='float32') / 255


def load_font(fontname, fontsize):
    return ImageFont.truetype(fontname, fontsize)
