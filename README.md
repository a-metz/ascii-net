# ascii-net

An experiment in generating ASCII Art from images with an artificial neural network.

_IN PROGRESS_

## Usage ##

Currently only a hard-coded test image is converted:

    $ python test_ocr.py

**Source:**

![source image](/test/test_image_w.png?raw=true)

**Generated:**

![screenshot](/docs/screenshot.png?raw=true)

## Implementation ##

This experiment trains a neural network model as an optical character recognizer.

The model is a simple 2 layer neural network:

* 99 inputs (glyphs in the font are 9x11 pixels)
* fully connected hidden layer with 99 nodes with bias and sigmoid activation
* fully connected output layer with 92 nodes (number of different characters used) with bias and softmax activation
* cross entrophy loss function

It is trained with rendered glyphs of a font and corresponding labels as inputs:

![a](/docs/glyph_a.png?raw=true) --> a
![b](/docs/glyph_b.png?raw=true) --> b
![#](/docs/glyph_hash.png?raw=true) --> #

And uses those to predict the best label for the tiles in an input image:

![_](/docs/tile_00_underline.png?raw=true) --> \_
![,](/docs/tile_01_comma.png?raw=true) --> ,
![#](/docs/tile_02_hash.png?raw=true) --> #
![*](/docs/tile_03_star.png?raw=true) --> \*
![7](/docs/tile_04_7.png?raw=true) --> 7

Two implementations of the model were made:

### NNet Model ###

This is my own implementation of MLPs using only numpy arrays. It supports batch learning with stochastic gradient descent.

Developing this has been a great help in understanding the maths behind the back propagation algorithm and why it is so efficient for calculating the loss derivates necessary for gradient descent. I highly recommend the blog post [Calculus on Computational Graphs: Backpropagation](http://colah.github.io/posts/2015-08-Backprop/) for further reading.

### Keras Model ###

This is an implementation using the [Keras Deep Learning Library](http://keras.io/). It is trained using stochastic gradient descent with Nesterov momentum.

Training this model is quite a bit faster then for my numpy implementation and really simple to implement.

## Future work ##

* Use deep convolutional neural nets to detect more abstract features for better selection of ASCII character
* generate more general training data by randomly transforming the glyphs (translation, scale, shear)
* use overlapping segments of input image to include surrounding pixels into selection of ASCII character

