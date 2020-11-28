from ConvolutionLayer import ConvolutionLayer
import numpy as np


def test_one_dim():
    input = np.ones(( 5, 5)).reshape((1,5,5))
    layer = ConvolutionLayer(1, (3, 3), random_weights=False)
    res = layer.forward_propagate(input)
    print(res)
    expected = np.array([[5,  7,  7,  7,  5],
                         [7, 10, 10, 10,  7],
                         [7, 10, 10, 10,  7],
                         [7, 10, 10, 10,  7],
                         [5,  7,  7,  7,  5]])
    print(expected)
    assert np.array_equal(res, expected)

def test_multidimensional_input():
    input = np.ones((2,5,5))
    layer = ConvolutionLayer(1, (3,3))
    res = layer.forward_propagate(input)
    expected = np.array([[[5,  7,  7,  7,  5],
                          [7, 10, 10, 10,  7],
                          [7, 10, 10, 10,  7],
                          [7, 10, 10, 10,  7],
                          [5,  7,  7,  7,  5]],
                         [[5,  7,  7,  7,  5],
                          [7, 10, 10, 10,  7],
                          [7, 10, 10, 10,  7],
                          [7, 10, 10, 10,  7],
                          [5,  7,  7,  7,  5]]])
    assert np.array_equal(res, expected)


