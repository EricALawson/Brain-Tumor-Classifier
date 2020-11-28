import numpy as np
from numpy.lib.stride_tricks import as_strided
from parameters import seed


class ConvolutionLayer:
    def __init__(self, num_filters, filter_dimensions, step=1, random_weights=True):
        np.random.seed(seed)
        self.filter_dimensions = filter_dimensions
        if random_weights == False:
            self.filters = np.ones((num_filters, *filter_dimensions))
            self.bias = np.ones(num_filters)
        else:
            self.filters = np.random.rand(num_filters, *filter_dimensions)
            self.bias = np.random.rand(num_filters)
        
        self.step = step

    def forward_propagate(self, activation_prev):
        print("forward")
        print(activation_prev.shape)
        pad_width = np.floor_divide(self.filter_dimensions,  2)
        # if activation_prev.shape[0] == 1:
        #     pads = (pad_width, pad_width)
        # else:
        pads = ((0,0), pad_width, pad_width)
        padded = np.pad(activation_prev, pad_width=pads)
        # print(padded)
        shape = activation_prev.shape + self.filter_dimensions
        # print(activation_prev.shape)
        # print(shape)
        strides = padded.strides + padded.strides[-2:]
        # print(strides)
        # a view of padded as an array of windows with dimension = filter_dimensions
        windows = as_strided(padded, shape=shape, strides=strides, writeable=False)
        # print(windows.shape)
        # print(self.filters.T.shape)
        #sums = np.einsum('nml,ijkmn->lijk', self.filters.T, windows)
        # cache z for back propagation
        sums = np.tensordot(windows, self.filters.T, axes=2)
        print(sums.shape)
        reordered = np.moveaxis(sums, -1, 0)
        biased = reordered + self.bias
        # print(self.z.shape)
        self.z = biased.reshape(biased.shape[0]*biased.shape[1], *biased.shape[2:]).squeeze()
        reLU = np.where(self.z > 0, self.z, 0)
        self.a = reLU # cache activation for back propagation
        return reLU


# t_arr = np.arange(64)
# arr1 = np.reshape(t_arr, (8,8))

# layer = ConvolutionLayer(1,(1,1))
# print(layer.forward_propagate(arr1))
 
 
