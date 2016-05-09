import numpy as np

from PIL import Image, ImageDraw, ImageFont


class GlyphList(list):
	def __init__(self, charset, fontname, fontsize, dim):
		list.__init__(self)

		self.charset = charset
		self.num_pixels = dim[0] * dim[1]

		font = ImageFont.truetype(fontname, fontsize)

		# generate glyphs
		for index, char in enumerate(charset):
			g = Glyph()
			g.index = index
			g.char = char
			g.image = generate_image(char, font, dim)
			g.array = np.asarray(g.image, dtype='float64').flatten() / 255

			self.append(g)


class Glyph:
	pass


def generate_image(char, font, dim):
	img = Image.new('L', dim, color=255)
	dh = ImageDraw.Draw(img)
	dh.text((0, 0), char, font=font, fill=0)
	return img


def generate_default_glyphs():
	return GlyphList(	
		charset='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;?@[\\]^_`{|}~ ',
		fontname='fonts/DejaVuSansMono.ttf',
		fontsize=18,
		dim=(11, 23))
