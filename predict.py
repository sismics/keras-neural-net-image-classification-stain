import numpy as np
from PIL import Image, ImageDraw
from keras.models import load_model
from keras.preprocessing import image

# dimensions of our images
img_width, img_height = 64, 64
input_image = 'image2.jpg'
input_model = 'model.h5'
dpi_in = 600
dpi_out = 300


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
            yield im.crop(box)


# load the model we saved
model = load_model(input_model)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# slice the input image
images_array = []
ignored = {}
for k, piece in enumerate(crop(input_image, img_width, img_height), 0):
    # transform the image in an array on 0. to 1.
    img_array = image.img_to_array(piece.convert('L'))
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255
    if np.var(img_array) < 0.001:
        ignored[k] = True
    images_array.append(img_array)


# pass the list of multiple images np.vstack()
images = np.vstack(images_array)
classes = model.predict(images, batch_size=10)

# draw the grid
output = load_dpi(input_image).convert('RGBA')
tmp = Image.new('RGBA', output.size, (0, 0, 0, 0))
d = ImageDraw.Draw(tmp)

p = 0
for m in range(output.height // img_height):
    for n in range(output.width // img_width):
        print(classes[p][0])
        if p in ignored:
            d.rectangle([n * img_width, m * img_height, (n + 1) * img_width, (m + 1) * img_width], fill=(0, 255, 0, 64))
        elif classes[p][0] > 0.6:
            d.rectangle([n * img_width, m * img_height, (n + 1) * img_width, (m + 1) * img_width],
                        fill=(255, 0, 0, int(64 * (classes[p][0] * 2 - 0.5))))
        p += 1

output = Image.alpha_composite(output, tmp)
output.save('output.png')
