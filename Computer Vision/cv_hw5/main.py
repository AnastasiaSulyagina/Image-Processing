from __future__ import division
import struct
from array import array
import numpy as np

test_img_path = 't10k-images.idx3-ubyte'
test_lbl_path = 't10k-labels.idx1-ubyte'
train_img_path = 'train-images.idx3-ubyte'
train_lbl_path = 'train-labels.idx1-ubyte'


def load(path_img, path_lbl):
    with open(path_lbl, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        labels = array("B", file.read())

    with open(path_img, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        image_data = array("B", file.read())

    images = []
    for i in range(size):
        images.append([0] * rows * cols)

    for i in range(size):
        images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

    lbl = []
    img = []
    for i in range(len(labels)):
        if labels[i] == 0:
            lbl.append(-1)
            img.append(images[i])
        elif labels[i] == 1:
            lbl.append(1)
            img.append(images[i])
    img, lbl = np.array(img), np.array(lbl)
    for i in img:
        i[i < 190] = 0
        i[i >= 190] = 1

    return img, lbl

train_images, train_labels = load(train_img_path, train_lbl_path)
test_images, test_labels = load(test_img_path, test_lbl_path)

w = np.zeros(784)


def grad(w):
    res = np.zeros(784)
    for i in range(len(train_images)):
        res += (train_labels[i] * train_images[i]) / (1 + np.exp(np.dot(train_labels[i], w.T) * train_images[i]))
    return (-1 / len(train_images)) * res

nu = 0.01
for t in range(200):
    w_c = w.copy()
    w -= nu * grad(w.copy())
print w
tp, tn, fp, fn = 0, 0, 0, 0
res = []
for i in range(len(test_images)):
    x = -np.dot(test_images[i], w.T)
    val = 1 / (1 + np.exp(x))
    print val
    res.append(1 if val > 0.5 else -1)
    if res[i] == test_labels[i]:
        if res[i] == 1:
            tp += 1
        else:
            tn += 1
    else:
        if res[i] == 1:
            fp += 1
        else:
            fn += 1

print tp, tn, fp, fn
print ("precision: " + str(tp / (tp + fp)))
print ("recall: " + str(tp / (tp + fn)))