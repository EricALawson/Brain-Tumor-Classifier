from ConvolutionLayer import ConvolutionLayer
from functools import reduce
from load_data import read_images
import numpy as np

class Classifier:
    def __init__(self, *layers):
        self.layers = layers

    def forward_propagate(self, image):
        predictions = reduce(
            lambda im, layer: layer.forward_propagate(im),
            self.layers,
            image
        )
        return predictions

X, Y = read_images()
oneX = np.array(X[0])


tumor_classifier = Classifier(
    ConvolutionLayer(2, (2, 2)),
    ConvolutionLayer(1, (2, 2))
)

out = tumor_classifier.forward_propagate(oneX)
 