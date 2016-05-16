import numpy as np

from PIL import Image, ImageDraw, ImageFont


def generate_image(char, font, dim):
    img = Image.new('L', dim, color=255)
    dh = ImageDraw.Draw(img)
    dh.text((0, 0), char, font=font, fill=0)
    return img


def generate_data(charset, fontname, fontsize, dim):
    num_datasets = len(charset)

    num_pixels = dim[0] * dim[1]
    num_classes = len(charset)

    inputs = np.zeros((num_datasets, num_pixels))
    labels = np.zeros((num_datasets, num_classes))

    font = ImageFont.truetype(fontname, fontsize)

    for index, char in enumerate(charset):
        inputs[index, :] = np.asarray(
            generate_image(char, font, dim),
            dtype='float32').flatten() / 255
        label = np.zeros(num_classes)
        label[index] = 1
        labels[index, :] = label

    return inputs, labels


def generate_default_data():
    return generate(
        charset=
        '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;?@[\\]^_`{|}~ ',
        fontname='fonts/DejaVuSansMono.ttf',
        fontsize=18,
        dim=(11, 23))