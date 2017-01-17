import zipfile
import os
import struct
import numpy as np

"""
Loads all training or testing samples.
Returns 1-D numpy arrays, to be input into neurons.
"""
def load(self, data = 'training', path = "."):
    labels, images = [], []

    if data is 'training':
        label_path = os.path.join(path, 'data/train-labels.idx1-ubyte')
        image_path = os.path.join(path, 'data/train-images.idx3-ubyte')
    else:
        label_path = os.path.join(path, 'data/t10k-labels.idx1-ubyte')
        image_path = os.path.join(path, 'data/t10k-images.idx3-ubyte')

    with open(label_path, 'rb') as label_file:
        struct.unpack(">II", label_file.read(8))
        labels = np.fromfile(label_file, dtype = np.uint8)

    with open(image_path, 'rb') as img_file:
        magic, size, rows, cols = struct.unpack(">IIII", img_file.read(16))
        images = np.fromfile(img_file, dtype = np.uint8).reshape(len(labels), rows , cols)

    for pos in range(len(labels)):
        images[pos].flatten()
        yield (labels[pos], images[pos].flatten() / 64.0)
