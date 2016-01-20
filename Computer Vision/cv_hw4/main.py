import cv2
import math
import random as rand
import numpy as np
from numpy import random
from numpy import linalg
from scipy.optimize import leastsq

#1

def gen_dots(n, x, y):
    dots = []
    for i in range(n):
        a, b = random.uniform(), random.uniform()
        dots.append(x * a + y * b)
    return np.array(dots)

def normalize(a):
    x, y, z = a
    norm = math.sqrt(x*x + y*y + z*z)
    return a / float(norm)

# main
def mse():
    def get_norm(points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        A = np.column_stack((x, y, np.ones(len(x))))
        At = A.T
        k = linalg.inv(At.dot(A)).dot(At).dot(z.reshape(len(points), 1))
        a, b, _ = k[0][0], k[1][0], k[2][0]

        return linalg.norm(normalize(np.array([a, b, -1])) - cross)

    u = normalize(np.array([3., 5., 7.]))
    v = normalize(np.array([5., 4., -5.]))
    cross = normalize(np.cross(u, v))

    points = gen_dots(100, u, v)
    points += np.random.normal(0, 0.1, [100, 3])

    out = open('result1.txt', 'w')
    for i in range(1, 10):
        out.write(str(get_norm(points[:i*10])) + '\n')
#mse()

#2

def new_mse():
    def norm(x):
        coef = x[2]
        return np.delete(x, 2, 0) / (0.00001 if coef == 0 else coef)

    def fun(x):
        rt, t = np.split(x, 2)
        t = np.transpose(t)
        par, _ = cv2.Rodrigues(rt)
        coords = map(lambda x: norm(K.dot(par.dot(np.insert(x, 2, np.float32(0.))) + t)), board_pts[0:n])
        return (coords - img_pts[0:n]).flatten()

    def get_err():
        sum = 0.
        for i in range(8):
            coords = norm(K.dot(R.dot(np.insert(board_pts[i], 2, np.float32(0.))) + t))
            sum += np.linalg.norm(coords - img_pts[i])
        return sum / 8

    img = cv2.imread('chessboard.png', 0)
    h, w = img.shape
    sz, n, f = 333, 8, 6741
    img_pts = np.float32([[228, 297], [342, 852], [790, 170], [913, 723],  # outers
                          [397, 402], [456, 680], [678, 339], [739, 615]])
    board_pts = sz * np.float32([[1, 1], [1, 5], [5, 1], [5, 5],  # outers
                                 [2, 2], [2, 4], [4, 2], [4, 4]])
    for x, y in img_pts:
        img[y][x] = np.float32(255)
    cv2.imwrite("img.jpg", img)

    # homography
    H, _ = cv2.findHomography(board_pts[0:4], img_pts[0:4])
    K = np.float32([[f, 0, w / 2],
                    [0, f, h / 2],
                    [0, 0, 1]])
    Kinv = linalg.inv(K)
    Hn = Kinv.dot(H)
    Hn /= linalg.norm(Hn[:, 0])

    # spin
    r1, r2, t = Hn[:, 0], Hn[:, 1], Hn[:, 2]
    r = np.column_stack((r1, r2, np.cross(r1, r2)))

    # parametrization -> Levenberg-Marquardt
    p, _ = cv2.Rodrigues(r)
    p = np.transpose(p)[0]
    p = np.concatenate((p, t))

    # errors
    out = open('result2.txt', 'w')
    for n in range(4, 9):
        p, _ = leastsq(fun, p)
        t = p[3:6]
        R, _ = cv2.Rodrigues(p[0:3])
        sum = get_err()
        out.write(str(sum) + '\n')
    out.close()
new_mse()

#3

def ransac_mse():
    def norm(x):
        coef = x[2]
        return np.delete(x, 2, 0) / (0.00001 if coef == 0 else coef)

    def fun(x):
        rt, t = np.split(x, 2)
        t = np.transpose(t)
        par, _ = cv2.Rodrigues(rt)
        coords = map(lambda x: norm(K.dot(par.dot(np.insert(x, 2, np.float32(0.))) + t)), max_pts[0:n])
        return (coords - max_img[0:n]).flatten()

    def get_err():
        sum = 0.
        for i in range(len(max_pts)):
            coords = norm(K.dot(R.dot(np.insert(max_pts[i], 2, np.float32(0.))) + t))
            sum += np.linalg.norm(coords - max_img[i])
        return sum / len(max_pts)

    img = cv2.imread('chessboard.png', 0)
    h, w = img.shape
    sz, n, f = 333, 8, 6741
    img_pts = np.concatenate((np.float32([[228, 297], [342, 852], [790, 170], [913, 723],
                                          [397, 402], [456, 680], [678, 339], [739, 615]]),
                              np.random.random_integers(0, min(h, w), (8, 2))))

    board_pts = sz * np.concatenate((np.float32([[1, 1], [1, 5], [5, 1], [5, 5],
                                                 [2, 2], [2, 4], [4, 2], [4, 4]]),
                                     np.random.random_integers(0, 6, (8, 2))))

    # N - RANSAC iterations
    # p - probability
    # s - amount of elements
    # e - % of bad points
    p, e, s = 0.99, 0.5, 4
    N = int(math.log(1 - p) / math.log(1 - math.pow(1 - e, s)))

    eps = 15
    max_pts, max_img, best_sample = [], [], []

    for i in range(N):
        sample = rand.sample(range(16), 4)
        new_H, _ = cv2.findHomography(board_pts[sample], img_pts[sample])

        if new_H is not None:
            coords = map(lambda x: norm(new_H.dot(np.insert(x, 2, np.float32(1)))), board_pts)
            dif = abs(coords - img_pts)
            dif = [i for i in range(len(coords)) if dif[i][0]**2 + dif[i][1]**2 < eps**2]
            coords = np.array(coords)
            pts, imgpts = coords[dif], img_pts[dif]

            if len(pts) > len(max_pts):
                max_pts, max_img, H, best_sample = pts, imgpts, new_H, sample

    K = np.float32([[f, 0, w / 2],
                    [0, f, h / 2],
                    [0, 0, 1]])
    Kinv = linalg.inv(K)
    Hn = Kinv.dot(H)
    Hn /= linalg.norm(Hn[:, 0])

    r1, r2, t = Hn[:, 0], Hn[:, 1], Hn[:, 2]
    r = np.column_stack((r1, r2, np.cross(r1, r2)))

    p, _ = cv2.Rodrigues(r)
    p = np.transpose(p)[0]
    p = np.concatenate((p, t))

    out = open('result3.txt', 'w')

    print len(max_pts)
    # 8 non-random points (with 0.99 probability)
    out.write("points: " + str([(x, y) for (x, y) in max_pts]))
    out.write("\n\nsample: " + str(best_sample))

    out.write("\n\nerrors:\n")
    for n in range(4, len(max_pts)):
        p, _ = leastsq(fun, p)
        t = p[3:6]
        R, _ = cv2.Rodrigues(p[0:3])
        sum = get_err()
        out.write(str(sum) + '\n')

    out.close()
ransac_mse()
