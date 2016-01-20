import cv2
import numpy as np
import math

# preparations
img = cv2.imread('hall405.jpg')
h, w, _ = img.shape

pts = np.float32([[507, 132], [629, 77], [506, 308], [631, 307]])
dst_pts = np.float32([[0, 0], [299, 0], [0, 299], [299, 299]])
points = [[515, 230], [530, 160], [580, 160], [610, 300]]

for x, y in points:
    img[y][x] = np.float32([0, 0, 255])
for x, y in pts:
    img[y][x] = np.float32([255, 0, 0])

# find homography + get view
A, _ = cv2.findHomography(pts, dst_pts, cv2.RANSAC, 5.0)
print A
dst = cv2.warpPerspective(img, A, (300, 300))
cv2.imwrite("view.jpg", dst)
cv2.imwrite("with_points.jpg", img)

# find reds on dest
reds = []

for i in range(300):
    for j in range(300):
        x, y, z = dst[i][j]
        if x < 35 and y < 45 and z > 185:
            l = len(reds)
            if not(l != 0 and abs(reds[l - 1][0] - j) + abs(reds[l - 1][1] - i) < 3):
                print x, y, z
                reds.append((j, i))

reds = sorted(reds)
out = open('result.txt', 'w')

# count error
for i in range(len(points)):
    a, b = points[i]
    c, d = reds[i]
    x, y, coef = np.dot(A, [a, b, 1])
    x, y = x/coef, y/coef
    err = math.sqrt((x - c) * (x - c) + (y - d) * (y - d))
    out.write(str(err) + '\n')


