import skimage
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
import os
import math
import numpy as np

import parameters

def read_images():
    X_yes, Y_yes = read_dir('yes')
    X_no, Y_no= read_dir('no')
    X = X_yes + X_no
    Y = Y_yes + Y_no
    image_details(X, Y)
    return resize_images(X), Y

def read_dir(dir_name):
    label_to_y = {
        'yes' : 1,
        'no' : 0
    }
    files = os.listdir(dir_name)
    X = []
    Y = []
    for file in files:
        #X.append( imread(dir_name + '/' + file, as_gray=parameters.grayscale) )
        X.append(Image.open(dir_name + '/' + file).convert('F'))
        Y.append( label_to_y[dir_name] )

    return X, Y
    
def resize_images(X):
    shapes = [shape.size for shape in X]
    heights = [shape[1] for shape in shapes]
    min_height = min(heights)

    def resize(im):
        new_height = min_height
        new_width = int(min_height * im.size[0] / im.size[1])
        return im.resize( (new_width, new_height) )

    resized = [resize(im) for im in X]
    #print([im.size for im in resized])

    resized_shapes = [shape.size for shape in resized]
    widths = [shape[0] for shape in resized_shapes]
    min_width = min(widths)

    def crop(im):
        diff = im.size[0] - min_width
        left = math.ceil(diff / 2)
        right = left + min_width
        top = 0
        bottom = im.size[1]
        return im.crop( (left, top, right, bottom) )

    cropped = [crop(im) for im in resized]
    #print([im.size for im in cropped])
    return cropped
    

    
def image_details(X, Y):
    print('total images: ', len(X) )

#read_images()
