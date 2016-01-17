import numpy as np
import cv2
import itertools

img = cv2.imread('Lena.jpg', -1)
h, w, c = img.shape

#a
LenaMixed = cv2.imread('Lena.jpg', -1)
left_corner = img[0:h/2, 0:w/2]
right_corner = img[h/2:h, w/2: w]
LenaMixed[0:h/2, 0:w/2] = right_corner
LenaMixed[h/2:h, w/2:w] = left_corner
cv2.imwrite('LenaMixed.jpg', LenaMixed)

#b
LenaGrayMean = cv2.imread('Lena.jpg',-1)
b, g, r = cv2.split(LenaGrayMean)

def f(x, y, z):
    md = []
    for i in range(0, w):
        val = (np.uint(x[i]) + np.uint(y[i]) + np.uint(z[i])) / 3
        md.append(np.ubyte(val))
    return md

mid = np.array(map(f, b, g, r))
LenaGrayMean = mid
cv2.imwrite('LenaGrayMean.jpg', LenaGrayMean)

#c
new_cwt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('Lenanew.jpg', new_cwt)
LenaMeanDifference = new_cwt - LenaGrayMean
cv2.imwrite('LenaMeanDifference.jpg', LenaMeanDifference)

#d
LenaMAXSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
LenaMAXSV[:,:,2] = 255
LenaMAXSV[:,:,1] = 255
LenaMAXSV = cv2.cvtColor(LenaMAXSV, cv2.COLOR_HSV2BGR)
cv2.imshow('LenaMAXSV.jpg',LenaMAXSV)
cv2.imwrite('LenaMAXSV.jpg',LenaMAXSV)

#e
b,g,r = cv2.split(img)
Lenabgr = cv2.imread('Lena.jpg',-1)
cv2.imwrite('Lenabgr.jpg',Lenabgr)
Lenabrg = cv2.merge((b,r,g))
cv2.imwrite('Lenabrg.jpg',Lenabrg)
Lenargb = cv2.merge((r,g,b))
cv2.imwrite('Lenargb.jpg',Lenargb)
Lenarbg = cv2.merge((r,b,g))
cv2.imwrite('Lenarbg.jpg',Lenarbg)
Lenagrb = cv2.merge((g,r,b))
cv2.imwrite('Lenagrb.jpg',Lenagrb)
Lenagbr = cv2.merge((g,b,r))
cv2.imwrite('Lenagbr.jpg',Lenagbr)

#f
LenaHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue, s, v = cv2.split(LenaHSV)
v = np.array(map(lambda x: 255 - x, v))
LenaInvertLuminance = cv2.merge((hue, s, v))
LenaInvertLuminance = cv2.cvtColor(LenaInvertLuminance, cv2.COLOR_HSV2BGR)
cv2.imwrite('LenaInvertLuminance.jpg',LenaInvertLuminance)

#g
LenaLab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

B, G, R = cv2.split(img)
# BGR to XYZ

def from_rgb_to_xyz(val):
    for norm_val in val:
        norm_val /= 255
        if norm_val > 0.04045:
            norm_val = ((norm_val + 0.055) / 1.055) ** 2.4
        else:
            norm_val /= 12.92
        norm_val *= 100
    return val

var_R = list(itertools.chain.from_iterable(map(from_rgb_to_xyz, R.tolist())))
var_G = list(itertools.chain.from_iterable(map(from_rgb_to_xyz, G.tolist())))
var_B = list(itertools.chain.from_iterable(map(from_rgb_to_xyz, B.tolist())))

X = map((lambda x, y, z: x * 0.4124 + y * 0.3576 + z * 0.1805), var_R, var_G, var_B)
Y = map((lambda x, y, z: x * 0.2126 + y * 0.7152 + z * 0.0722), var_R, var_G, var_B)
Z = map((lambda x, y, z: x * 0.0193 + y * 0.1192 + z * 0.9505), var_R, var_G, var_B)

# XYZ to CIELab
var_X = map(lambda x: x / 95.047, X)
var_Y = map(lambda y: y / 100.000, Y)
var_Z = map(lambda z: z / 108.883, Z)

def from_xyz_to_lab(val):
    if val > 0.008856:
        val **= (1/3)
    else:
        val = (7.787 * val) + (16 / 116)
    return val

L = map(from_xyz_to_lab, X)
A = map(from_xyz_to_lab, Y)
B = map(from_xyz_to_lab, Z)

def to_list_of_lists(lst):
    result = []
    for i in range(0, h):
        piece = []
        for j in range(0, w):
            piece.append(lst[i * h + j])
        result.append(piece)
    return result

l = to_list_of_lists(map(lambda x: (116 * x) - 16, A))
a = to_list_of_lists(map((lambda x, y: 500 * (x - y)), L, A))
b = to_list_of_lists(map((lambda x, y: 200 * (x - y)), A, B))

LenaLabManual = cv2.merge((np.array(l), np.array(a), np.array(b)))
# LenaLabDiff = LenaLab - LenaLabManual
# LenaLabDiff = cv2.cvtColor

LenaLab = cv2.cvtColor(LenaLab, cv2.COLOR_LAB2BGR)
LenaLabDiff = LenaLab - LenaLabManual
cv2.imwrite('LenaLab.jpg', LenaLab)
cv2.imwrite('LenaLabManual.jpg', LenaLabManual)
cv2.imwrite('LenaLabDiff.jpg', LenaLabDiff)
