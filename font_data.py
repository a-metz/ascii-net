from PIL import Image, ImageDraw, ImageFont
import numpy as np

class Glyph:
	def __init__(self, char, font, dim):
		self.char = char
		self.font = font
		self.dim = dim
		self.img = Image.new('L', dim, color=255)
		dh = ImageDraw.Draw(self.img)
		dh.text((0, 0), char, font=font, fill=0)
		self.imgarray = np.asarray(self.img, dtype='float64')

def generate_glyph_set(
		char_set='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;?@[\\]^_`{|}~ ',
		fontname='DejaVuSansMono.ttf',
		fontsize=18,
		dim=(11, 23)
		):
	glyph_set = []
	font = ImageFont.truetype(fontname, fontsize)
	for char in char_set:
		glyph_set.append(Glyph(char, font, dim))
	return glyph_set