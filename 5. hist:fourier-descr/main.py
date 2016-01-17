import numpy as np
import cv2
import os

def print_file(data, filename):
    output = open(filename, "w", buffering=11000000)
    str_data = [str(x) + " " + str(y) + " " + str(z) + "\n" for (x, y, z) in data]
    output.writelines(str_data)
    output.flush()
    output.close()

# distance
hists = []

def chi_sq(hist1, hist2, eps=1e-6):
    dist = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                         for (a, b) in zip(hist1, hist2)])
    return dist


def read_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #hist = cv2.calcHist(img, [0, 1, 2], None, [10, 7, 8], [0, 180, 0, 256, 0, 256])
    #hist = cv2.calcHist(img, [0, 1, 2], None, [6, 7, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.calcHist(img, [0, 1, 2], None, [3, 4, 10], [0, 180, 0, 256, 0, 256])
    hists.append((os.path.basename(path), hist))


def count_dist(dir):
    chi_res = []
    l2_res = []

    for name in os.listdir(dir):
        path = os.path.join(dir, name)
        read_img(path)

    for i in range(0, len(hists)):
        for j in range(i + 1, len(hists)):
            name1, hist1 = hists[i]
            name2, hist2 = hists[j]
            chi = (name1, name2, chi_sq(hist1, hist2))
            chi_res.append(chi)
            l2 = (name1, name2, cv2.norm(hist1, hist2, cv2.NORM_L2))
            l2_res.append(l2)

    chi_res = sorted(chi_res, key=lambda x: x[2])
    print_file(chi_res, "chi_square.txt")
    l2_res = sorted(l2_res, key=lambda x: x[2])
    print_file(l2_res, "l2.txt")

count_dist("Corel")

# descriptors


descriptors = []

def get_descr(path):
    n = 7000
    img = cv2.imread(path)
    h, w, _ = img.shape

    img = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cx, cy = h/2, w/2
    contour, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = contour[0][:, 0, :]

    for i in contour:
        c = i[:, 0, :]
        points = np.vstack((points, c))

    s = []
    for (x, y) in points:
        s.append(np.sqrt((x - cx) ** 2 + (y - cy) ** 2))

    s = np.array(s)
    fs = abs(cv2.dft(s))
    fs = fs / fs[0]

    step = n/len(fs) + 1 if len(fs) <= n else 1
    f_descr = np.repeat(fs, step)
    f_descr = f_descr[:n]
    descriptors.append((os.path.basename(path), f_descr))


def count_descr(dir):
    l2_res = []

    for name in os.listdir(dir):
        path = os.path.join(dir, name)
        get_descr(path)

    for i in range(0, len(descriptors)):
        for j in range(i + 1, len(descriptors)):
            name1, descr1 = descriptors[i]
            name2, descr2 = descriptors[j]
            l2 = (name1, name2, cv2.norm(descr1, descr2, cv2.NORM_L2), )
            l2_res.append(l2)

    l2_res = sorted(l2_res, key=lambda x: x[2])
    print_file(l2_res, "leaves.txt")

#count_descr("leaves")
