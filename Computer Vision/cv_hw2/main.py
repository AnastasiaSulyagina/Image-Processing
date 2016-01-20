# coding=utf-8
import numpy as np
import cv2
import math

def affine_remap(x, y, h, w, xx, yy, hh, ww):
    img = cv2.imread("lena.jpg", -1)
    map_x = np.zeros(img.shape[:2],np.float32)
    map_y = np.zeros(img.shape[:2],np.float32)
    mvx, mvy = xx - x, yy - y
    xx -= mvx
    hh -= mvx
    yy -= mvy
    ww -= mvy
    scx, scy = (h - x) / float(hh - xx), (w - y) / float(ww - yy)
    for j in range(h):
        for i in range(w):
            map_x.itemset((j, i), scx * (i - mvx))
            map_y.itemset((j, i), scy * (j - mvy))

    dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return dst

def affine_warp(x, y, h, w, xx, yy, hh, ww):
    img = cv2.imread("lena.jpg", -1)
    pts1 = np.float32([[x, y], [h, w], [x, w]])
    pts2 = np.float32([[xx, yy], [hh, ww], [xx, ww]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img, M, (h, w))
    return dst


def affine_both():
    img = cv2.imread("lena.jpg", -1)
    x, y, x1, y1 = 0, 0, img.shape[0], img.shape[1]
    x2, y2, x3, y3 = 50, 200, img.shape[0], img.shape[1]
    img_remap = affine_remap(x, y, x1, y1, x2, y2, x3, y3)
    img_warpaffine = affine_warp(x, y, x1, y1, x2, y2, x3, y3)
    cv2.imwrite('remap1.png', img_remap)
    cv2.imwrite('wapraffine1.png', img_warpaffine)

affine_both()

def lines():
    #img = np.zeros((100, 100, 3), np.uint8)
    #img[:] = (255, 255, 255)
    img = cv2.imread("img.jpg", 0)
    h, w = img.shape
    pts = []

    for x in range(h):
        for y in range(w):
            if img[x][y] == 0:
                pts.append((y, x, 1))

    lines = [np.cross(pts[i], pts[j]) for i in range(len(pts)) for j in range(i + 1, len(pts))]

    for line in lines:
        a, b, c = line
        x1, x2 = -50, 150 # max img 100*100 => these xs are always out => line will cover all the img
        y1, y2 = -(a * x1 + c) / b, -(a * x2 + c) / b
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0))

    red_points = [(i / float(k), j / float(k))
                  for i, j, k in [np.cross(lines[l], lines[m])
                        for l in range(len(lines)) for m in range(l + 1, len(lines))]]

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for point in red_points:
        j, i = point
        if i >= 0 and j >= 0 and i < w and j < h:
            cv2.circle(img, (int(j), int(i)), 2, (0, 0, 255), -1)
    cv2.imwrite('imgwithlines.jpg', img)

#lines()

def third():
    def draw_lines(image, points):
        cv2.line(image, points[0], points[1], [0, 0, 0])
        cv2.line(image, points[1], points[3], [0, 0, 0])
        cv2.line(image, points[3], points[2], [0, 0, 0])
        cv2.line(image, points[2], points[0], [0, 0, 0])
        return image

    def perspective(point, cx, angle):
        x, y, z = point
        f = (cx * 1.0) / math.tan(math.radians(angle))
        return int(f * ((x * 1.0) / z) + cx), int(f * ((y * 1.0) / z) + cx)

    def orthographic(point, H, W):
        x, y, z = point
        return int(x * H / 5.0 + H / 2), int(y * W / 5.0 + W / 2)

    w = h = 1000
    img_pers = np.ones((h, w), np.uint8) * 255
    img_ort = np.ones((h, w), np.uint8) * 255

    # строю по клеткам, берем сторону квадрата за 3*sqrt(2) ед, расстояние камеры от плоскости квадрата 2*sqrt(2) ед
    # ед*sqrt(2) = клетка по диагонали.
    # Найдем местоположение точек
    pts = [(math.sqrt(2) * 3 / float(2), 1, 3), (-1 * math.sqrt(2) * 3 / float(2), 1, 3), (math.sqrt(2) * 3 / float(2), -2, 6),
           (-1 * math.sqrt(2) * 3 / 2, -2, 6)]

    pts_pers = [perspective(point, h / 2, 45) for point in pts]
    pts_ort = [orthographic(point, h, w) for point in pts]
    
    img_pers = draw_lines(img_pers, pts_pers)
    img_ort = draw_lines(img_ort, pts_ort)

    cv2.imwrite('perspective.jpg', img_pers)
    cv2.imwrite('orthographic.jpg', img_ort)
#third()
