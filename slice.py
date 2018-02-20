import os

import numpy as np
from PIL import Image

# dimensions of our images
from keras.preprocessing import image

img_width, img_height = 64, 64
dpi_in = 600
dpi_out = 300
directory = 'tmp/'


# load image at the right dpi
def load_dpi(infile):
    im = Image.open(infile)
    ratio = dpi_out / dpi_in
    return im.resize((int(im.width * ratio), int(im.height * ratio)), Image.ANTIALIAS)


# image cropping
def crop(infile, height, width):
    im = load_dpi(infile)
    for i in range(im.height // height):
        for j in range(im.width // width):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            s = im.crop(box)
            img_array = image.img_to_array(s.convert('L'))
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255
            if np.var(img_array) >= 0.001:
                yield s


count = 0
for filepath in os.listdir(directory):
    # slice the input image
    for k, piece in enumerate(crop(directory + filepath, img_width, img_height), 0):
        piece.save(directory + 'slice_' + str(count) + '.jpg')
        count += 1
