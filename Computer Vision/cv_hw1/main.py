import numpy as np
import cv2
import math


def make_projection():
    def ctg(x):
        return math.cos(x) / math.sin(x)
    h, w, angle, a_x, a_y, a_z = 600, 600, 60, -1, 1, 6
    angle /= 2
    f = h / 2 * ctg(angle * math.pi / 180.0)

    c_x, c_y = h / 2, w / 2
    ax, ay = f * a_x / a_z + c_x, f * a_y / a_z + c_y
    print(ax, ay)

make_projection()

def more_contrast(name):
    img = cv2.imread(name, 0)
    h, w = img.shape

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    black_point = 0
    white_point = 256
    for i in range(256):
        if hist[i] > 60:
            black_point = i
            break

    for i in reversed(range(256)):
        if hist[i] > 60:
            white_point = i
            break

    print black_point
    print white_point
    for i in range(h):
        for j in range(w):
            x = img[i][j]
            img[i][j] = np.ubyte(((255 * np.uint((x if x <= white_point else white_point) - black_point)) / np.uint(white_point - black_point)))

    cv2.imwrite('high-contrast.png', img)

#more_contrast('low-contrast.png')

def face_detection():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    img = cv2.imread('faces.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey+eh), (0, 255, 0), 2)

    cv2.imwrite('faces_detected.jpg', img)

#face_detection()
