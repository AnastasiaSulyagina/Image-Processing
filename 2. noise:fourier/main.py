import numpy as np
import cv2
import random
import math

def gradation(name='Lena.jpg'):
    a, b = 2, 100
    man = cv2.imread(name, -1)
    aut = cv2.imread(name, -1)
    h, w, _ = man.shape

    for i in range(h):
        for j in range(w):
            for k in range(3):
                x = (a * int(man[i][j][k]) + b)
                man[i][j][k] = x if x < 256 else 255

    aut = cv2.convertScaleAbs(aut, aut, a, b)

    cv2.imwrite('manualgradation.jpg', man)
    cv2.imwrite('autogradation.jpg', aut)
gradation()


def saltpepper(img, p, q):
    h, w = img.shape

    for i in range(h):
        for j in range(w):
            is_salt = random.random()
            is_pepper = random.random()
            img[i][j] = 255 if is_salt < p else 0 if is_pepper < q else img[i][j]
    return img

def splena():
    pq = [0.05, 0.1, 0.15]
    img = cv2.imread('Lena.jpg', -1)
    for k in range(3):
        b, g, r = cv2.split(img)
        new_img = cv2.merge((saltpepper(b, pq[k], pq[k]), saltpepper(g, pq[k], pq[k]), saltpepper(r, pq[k], pq[k])))
        cv2.imwrite(str(pq[k]) + 'saltpepper.jpg', new_img)
splena()

def gauss(img, mean, stdev):
    h, w = img.shape
    noise = stdev * np.random.randn(h, w) + mean

    for i in range(h):
        for j in range(w):
            if img[i][j] + noise[i][j] > 255:
                img[i][j] = 255
            elif img[i][j] + noise[i][j] < 0:
                img[i][j] = 0
            else:
                img[i][j] += noise[i][j]
    return img

def glena():
    pars = [[0, 30], [0, 60], [50, 30]]
    img = cv2.imread('Lena.jpg', -1)
    for k in range(3):
        b, g, r = cv2.split(img)
        new_img = cv2.merge((gauss(b, pars[k][0], pars[k][1]),
                             gauss(g, pars[k][0], pars[k][1]),
                             gauss(r, pars[k][0], pars[k][1])))
        cv2.imwrite(str(pars[k][0]) + '-' + str(pars[k][1]) + '-' + 'gauss.jpg', new_img)
glena()

def filtering(d):
    def apply_mask_and_DFT(mat, mask):
        fshift = mat * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        return cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    def write_final_img(d, pref, image, mag_spectrum, m, img_back):
        final_img = np.zeros((rows, (cols * 4)), np.uint8)
        final_img[0:rows, 0:cols] = image
        final_img[0:rows, cols:2 * cols] = mag_spectrum
        final_img[0:rows, 2 * cols:3 * cols] = m
        final_img[0:rows, 3 * cols:] = img_back
        cv2.imwrite(str(d) + pref + '.jpg', final_img)

    img = cv2.imread('Lena.jpg', 0)
    rows, cols = img.shape
    crow, ccol = rows/2, cols/2

    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    for i in range(rows):
        for j in range(cols):
            if magnitude_spectrum[i][j] > 255:
                magnitude_spectrum[i][j] = 255

    y,x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    mask = x*x + y*y <= d*d
    low_mask = np.zeros((rows, cols, 2), np.uint8)
    low_mask[mask] = 1
    high_mask = np.ones((rows, cols, 2), np.uint8)
    high_mask[mask] = 0

    # apply masks and inverse DFT
    low_img_back = apply_mask_and_DFT(dft_shift, low_mask)/(rows * cols)
    high_img_back = apply_mask_and_DFT(dft_shift, high_mask)/(rows * cols)

    low_m = magnitude_spectrum * cv2.split(low_mask)[0]
    high_m = magnitude_spectrum * cv2.split(high_mask)[0]

    write_final_img(d, 'low', img, magnitude_spectrum, low_m, low_img_back)
    write_final_img(d, 'high', img, magnitude_spectrum, high_m, high_img_back)
filtering(5)

def mse_one_channel(img1, img2):
    h, w = img1.shape
    sum = 0
    for i in range(h):
        for j in range(w):
                sum += int(int(img1[i][j]) - int(img2[i][j]))**2
    return sum/(h*w)

def MSE(img1, img2):
    sum = 0
    channels1 = cv2.split(img1)
    channels2 = cv2.split(img2)
    for k in range(3):
        sum += mse_one_channel(channels1[k], channels2[k])
    print sum/3
    return sum/3

def clean_diagonals(d):
    def apply_mask_and_DFT(mat, mask):
        fshift = mat * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        return cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    def write_final_img(d, pref, image, mag_spectrum, m, img_back):
        final_img = np.zeros((rows, (cols * 4)), np.uint8)
        final_img[0:rows, 0:cols] = image
        final_img[0:rows, cols:2 * cols] = mag_spectrum
        final_img[0:rows, 2 * cols:3 * cols] = m
        final_img[0:rows, 3 * cols:] = img_back
        cv2.imwrite(str(d) + pref + '.jpg', final_img)

    img = cv2.imread('Lena_diagonal.jpg', 0)
    rows, cols = img.shape
    crow, ccol = rows/2, cols/2

    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    for i in range(rows):
        for j in range(cols):
            if magnitude_spectrum[i][j] > 255:
                magnitude_spectrum[i][j] = 255

    y,x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    mask = x*x + y*y <= d*d
    low_mask = np.zeros((rows, cols, 2), np.uint8)
    low_mask[mask] = 1

    # apply masks and inverse DFT
    low_img_back = apply_mask_and_DFT(dft_shift, low_mask)/(rows * cols)

    low_m = magnitude_spectrum * cv2.split(low_mask)[0]

    write_final_img(d, 'lowdiag', img, magnitude_spectrum, low_m, low_img_back)
    print mse_one_channel(low_img_back, cv2.imread('lena_gray_512.tif', 0))
clean_diagonals(71)

def reduce_noise(img):
    h, w, _ = img.shape
    dir = [[-1, -1], [-1, 0], [0, -1], [0, 0], [0, 1], [1, 0], [1, 1]]

    for i in range(h):
        for j in range(w):
            for k in range(3):
                med = []
                for x, y in dir:
                     if i + x >= 0 and i + x < h and j + y >= 0 and j + y < w:
                        med.append(img[i + x][j + y][k])
                mid = len(med)/2
                med.sort()
                img[i][j][k] = med[mid]

    cv2.imwrite('lenamed.jpg', img)
    return img

def experiments():
    # MyMedian 1 + :
    # - : 220, MyMedian 2 : 181, Median 5 : 169, Gauss 5 5 : 158
    # MyMedian 2 + Gauss : 172, MyMedian 2 + Median : 176, Gauss + MyMedian 1 : 156
    # Bilat : 182, Median 1 + 5 : 167, Gauss 3 3: 158, Median 1 + Gauss 3: 162

    # Bilateral 8 60 60 : 134
    lena = cv2.imread('lena_color_512.tif', -1)
    my_lena = cv2.imread('lena_color_512-noise.tif', -1)
    reduce_noise(my_lena)
    bl = cv2.bilateralFilter(my_lena, 8, 60, 60)

    cv2.imwrite('lenamed.jpg', bl)
    MSE(bl, lena)
experiments()

def sv(img, n):
    img = cv2.imread(img, 0)
    mat = np.ones((n, n), np.float) / float(n)
    img = cv2.filter2D(img, -1, mat)
    return img

def last():
    img1 = sv('Lena.jpg', 3)
    img2 = sv('Lena.jpg', 5)
    h, w = img1.shape

    for i in range(h):
        for j in range(w):
            img1[i][j] = abs(int(img1[i][j]) - int(img2[i][j]))
    cv2.imwrite('conv_diff.jpg', img1)
last()