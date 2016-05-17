# ascii-net

An experiment in generating ASCII Art from images with an artificial neural network.

_IN PROGRESS_

## Usage ##

Currently only a hard-coded test image is converted:

    $ python test_keras_ocr.py

**Source:**

![source image](/input_image/test_image_w.png?raw=true)

**Generated:**

![screenshot](/docs/screenshot.png?raw=true)

## Implementation ##

Current implementation uses a simple 2 layer neural network in keras:

* 99 inputs (glyphs in the font are 9x11 pixels)
* fully connected hidden layer with 99 nodes and sigmoid activation
* fully connected output layer with 92 nodes (number of different characters used) and softmax activation

It is trained using cross entropy as loss function and static gradient descent with Nesterov momentum.

## Future work ##

* Use deep convolutional neural nets to detect more abstract features for better selection of ASCII character
* generate more general training data by randomly transforming the glyphs (translation, scale, shear)
* use overlapping segments of input image to include surrounding pixels into selection of ASCII character
* complete implementation of neural network in numpy

