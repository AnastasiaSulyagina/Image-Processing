import os
import cv2
import numpy as np

images = 'imgs'
horizontal_scale = 31.
vertical_scale = 193.

def check_sin(img):
    h, w = img.shape
    line = img[h/2, :]
    zeros = len(line) - np.count_nonzero(line)
    if zeros < 8:
        return False
    else:
        return True


def check_pol(img):
    h, w = img.shape

    left = img[h/5, :w/2]
    right = img[h/5, w/2:]

    left_t = img[h/5 - 3, :w/2]
    right_t = img[h/5 - 3, w/2:]

    z_left = len(left) - np.count_nonzero(left)
    z_right = len(right) - np.count_nonzero(right)
    z_left_t = len(left_t) - np.count_nonzero(left_t)
    z_right_t = len(right_t) - np.count_nonzero(right_t)

    if max(z_left, z_left_t) > 0 and max(z_right, z_right_t) > 0:
        return True
    else:
        return False


def check_lin(img):
    h, w = img.shape

    left = img[h/5, :w/2]
    right = img[h/5, w/2:]

    left_t = img[h/5 - 3, :w/2]
    right_t = img[h/5 - 3, w/2:]

    z_left = len(left) - np.count_nonzero(left)
    z_right = len(right) - np.count_nonzero(right)
    z_left_t = len(left_t) - np.count_nonzero(left_t)
    z_right_t = len(right_t) - np.count_nonzero(right_t)

    if max(z_left, z_left_t) == 0 and max(z_right, z_right_t) >= 0:
        return True
    else:
        return False


def define_shape(img):
    h, w, _ = img.shape
    _, bw = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 80 * 255, 180, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    g = g / 80 * 255
    b = b / 80 * 255
    r = r / 80 * 255
    _, r = cv2.threshold(r, 180, 255, cv2.THRESH_BINARY)
    _, g = cv2.threshold(g, 180, 255, cv2.THRESH_BINARY)
    _, b = cv2.threshold(b, 180, 255, cv2.THRESH_BINARY)
    red = (g | b)
    green = (r | b)
    blue = (r | g)
    sin, pol, lin = bw, bw, bw
    if check_sin(red):
        sin = red
        if check_pol(green):
            pol = green
            lin = blue
        elif check_pol(blue):
            pol = blue
            lin = green
    else:
        if check_sin(green):
            sin = green
            if check_pol(red):
                pol = red
                lin = blue
            elif check_pol(blue):
                pol = blue
                lin = red
        else:
            if check_sin(blue):
                sin = blue
                if check_pol(green):
                    pol = green
                    lin = red
                elif check_pol(red):
                    pol = red
                    lin = green

    return sin, pol, lin

def get_sin_coef(img):
    a, b = 0, 0
    h, w = img.shape
    tst = img[h/2, :]
    fl, l, r = 0, -1, -1
    for i in range(w):
        if fl == 0 and tst[i] == 0:
            l = i
        if fl == 1 and tst[i] == 0:
            r = i
            break
        if l != -1 and tst[i] != 0:
            fl = 1
    period = r - l
    #print period

    a = 3.14 / (float(period) / horizontal_scale)

    for i in range(w/2 + 5, w):
        if tst[i] == 0:
            b = period - (i - w/2)
            break

    return a, float(b) / horizontal_scale + 0.16

def get_pol_coef(img):
    a, b, c, y, x, pt = 0, 0, 0, 0, 0, 0
    h, w = img.shape
    t, b, mid = 0, 0, h/2
    i = 0
    for i in reversed(range(h/2 - 5)):
        tst = img[i, :]
        if np.count_nonzero(tst) != len(tst):
            break

    y = h/2 - i
    bottom = img[i, :]
    l, r = 0, 0
    for j in range(w):
        if bottom[j] == 0:
            l = j
            break
    for j in reversed(range(w)):
        if bottom[j] == 0:
            r = j
            break
    x = w / 2 - (r + l) / 2

    ln = img[:, x + horizontal_scale * 7]
    for j in range(w / 2):
        if ln[j] == 0:
            pt = j
            break

    tst = img[:, w/2]
    u = 0
    for u in range(h/2):
        if tst[u] == 0:
            break
    x /= horizontal_scale
    y /= vertical_scale
    pt /= vertical_scale
    print x, y

    a = (pt - y) / 49.0
    c = (h/2 - u) / vertical_scale
    b = x * float(a)
    return a * 0.6, b, c - 0.01

def get_lin_coef(img):
    a, b = 0, 0
    h, w = img.shape
    i = 0
    tst = img[:, w/2]
    for i in range(h/2):
        if tst[i] == 0:
            break
    b = h/2 - i - 0.7
    pt = 0
    ln = img[:h/2, w/2 + (horizontal_scale * 7)]

    for j in reversed(range(h / 2 - 5)):
        if ln[j] == 0:
            pt = j
            break
    pt = h/2 - pt
    a = ((pt - b) / vertical_scale) / 7.0
    return a, b / vertical_scale

def get_coefs():
    a, b, c, d, e, f, g = 0, 0, 0, 0, 0, 0, 0
    with open("parameters.txt","w") as out:
        for o in os.listdir(images):
            print o
            imagePath = os.path.join(images, o)
            img = cv2.imread(imagePath)
            img = img[156:540, 102:719]
            sin, pol, lin = define_shape(img)

            a, b = get_sin_coef(sin)
            c, d, e = get_pol_coef(pol)
            f, g = get_lin_coef(lin)
            cv2.imwrite("sin" + o, sin)
            cv2.imwrite("pol" + o, pol)
            cv2.imwrite("lin" + o, lin)
            print a, b, c, d, e, f, g
            out.write(o + ', ' + str(a) + ', ' + str(b) + ', ' + str(c) + ', ' + str(d) + ', ' + str(e) + ', ' + str(f) + ', ' + str(g) + '\n')

get_coefs()






def crop():
    a, b, c, d, e, f = 0, 0, 0, 0, 0, 0
    for o in os.listdir(images):
        print o
        imagePath = os.path.join(images, o)
        img = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2GRAY)
        h, w = img.shape
        img[h/4 + 6, :] = 100
        # upper_thing = 156
        img[9 * h/10 - 1, :] = 100
        # bottom_thing = 539
        img[:, 9 * w/10] = 100
        # right_thing = 719
        img[:, w/8] = 100
        # left_thing = 102
        # print w/8
        cv2.imwrite("h" + o, img)
