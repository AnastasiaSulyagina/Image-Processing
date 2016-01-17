import cv2
import numpy as np

#1
def with_gradient(img):
    sobel_x = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3))
    sobel_y = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3))
    gradient = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    return gradient
    #cv2.imwrite('1res_gradient.jpg', gradient)

def with_laplacian(img):
    laplacian = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F))
    cv2.imwrite('1res_laplassian.jpg', laplacian)

def with_morph(img):
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite('1res_morphological.jpg', morph)

def find_edge():
    img = cv2.imread('portrait.jpg', 0)
    with_gradient(img)
    with_laplacian(img)
    with_morph(img)

#find_edge()

#2
def get_components(background):
    background *= 255
    img = cv2.imread('bin.png', 0)
    h, w = img.shape
    res = np.zeros((h + 1, w + 1), np.uint8)

    parent = np.zeros(100005)
    reds = np.random.randint(0, 255, 100005)
    greens = np.random.randint(0, 255, 100005)
    blues = np.random.randint(0, 255, 100005)
    reds[0] = 0
    greens[0] = 0
    blues[0] = 0

    cnt = 0
    parent[0] = 0

    def get_parent(a):
        start = a
        while a != parent[a]:
            a = parent[a]
        parent[start] = a
        return parent[a]

    def unite(a, b):
        parent[get_parent(b)] = get_parent(a)

    def are_equal(a, b):
        return get_parent(a) == get_parent(b)

    #1
    for i in range(h):
        for j in range(w):
            if img[i, j] != background:
                if res[i][j - 1] == 0 and res[i - 1, j] == 0:
                    cnt += 1
                    res[i][j] = cnt
                    parent[cnt] = cnt
                elif res[i][j - 1] != 0 and res[i - 1, j] != 0:
                    if not are_equal(res[i][j - 1], res[i - 1, j]):
                        unite(res[i][j-1], res[i-1, j])
                    res[i][j] = res[i][j - 1]
                elif res[i][j-1] != 0:
                    res[i][j] = res[i][j-1]
                else:
                    res[i][j] = res[i-1][j]
    #2
    blue = np.copy(img)
    green = np.copy(img)
    red = np.copy(img)

    for i in range(h):
        for j in range(w):
            blue[i][j] = blues[get_parent(res[i][j])]
            red[i][j] = reds[get_parent(res[i][j])]
            green[i][j] = greens[get_parent(res[i][j])]

    res = cv2.merge((blue, green, red))
    cv2.imwrite('2res_components.jpg', res)
#get_components(0)

#3
def erase_text():
    img = 255 - cv2.imread("table.jpg")

    kernel_h = np.zeros((15, 15), np.uint8)
    kernel_v = np.zeros((15, 15), np.uint8)
    kernel_rc = np.zeros((15, 15), np.uint8)
    kernel_rlc = np.zeros((15, 15), np.uint8)
    kernel_lc = np.zeros((15, 15), np.uint8)
    kernel_lrc = np.zeros((15, 15), np.uint8)

    kernel_h[7, :] = 1
    kernel_v[:, 7] = 1
    kernel_rc[7, 7:] = 1
    kernel_rc[1:7, 7] = 1
    kernel_rlc[7, 7:] = 1
    kernel_rlc[7:, 7] = 1
    kernel_lc[7, 1:7] = 1
    kernel_lc[7:, 7] = 1
    kernel_lrc[7, 1:7] = 1
    kernel_lrc[1:7, 7] = 1

    img_h = cv2.erode(img, kernel_h)
    img_v = cv2.erode(img, kernel_v)
    img_rc = cv2.dilate(cv2.erode(img, kernel_rc), kernel_lc)
    img_lc = cv2.dilate(cv2.erode(img, kernel_lc), kernel_rlc)
    img_lrc = cv2.erode(img,kernel_lrc)
    img_lrc = cv2.dilate(img_lrc, kernel_rlc)

    kernel_1 = np.ones((15, 15), np.uint8)
    res = 255 - cv2.morphologyEx(img_v + img_h + img_rc + img_lc + img_lrc, cv2.MORPH_CLOSE, kernel_1)

    cv2.imwrite("3res_table.jpg", res)
#erase_text()

#4 some lost circles can be seen but it's because of the radius, not algorithm
def process_circles():
    img = cv2.imread("circles.jpg", 0)
    h, w = img.shape
    help = np.copy(img)
    img_new = np.copy(img)
    borders = 255 - np.zeros((h + 40, w + 40), np.uint8)
    borders[20:h+20, 20:w+20] = img

    help = 255 - help
    kernel = np.zeros((14, 14), np.uint8)
    cv2.circle(kernel, (7, 7), 7, 1, -1)
    k = np.ones((18, 18), np.uint8)
    cv2.circle(k, (9, 9), 9, 0, -1)

    img = cv2.erode(img, kernel)
    help = cv2.erode(help, k)

    img &= help
    img = cv2.dilate(img, kernel)

    _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    cv2.imwrite("4res_single_circles.jpg", img)

    kernel_s = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel_s)

    img_new &= 255 - img
    img_new = cv2.erode(img_new, kernel_s)
    img_new = cv2.dilate(img_new, kernel_s)
    cv2.imwrite("4res_grouped_circles.jpg", img_new)

    k_border = np.ones((12, 12), np.uint8)
    cv2.circle(kernel, (6, 6), 6, 0, -1)

    borders = 255 - borders
    borders = cv2.erode(borders, kernel_s)
    borders = cv2.dilate(borders, k)
    borders = 255 - borders
    borders = cv2.dilate(borders, kernel)

    cv2.imwrite("4res_border_circles.jpg", borders)
#process_circles()

#5
def sort_coins():
    img = cv2.imread("coins_1.jpg", 0)
    img = cv2.medianBlur(img, 5)
    img = 255 - cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 20)

    kernel = np.zeros((7, 7), np.uint8)
    cv2.circle(kernel, (3, 3), 1, 255, -1)
    img = cv2.dilate(img, np.ones((3, 3), np.uint8))

    circles = cv2.HoughCircles(255 - img, cv2.cv.CV_HOUGH_GRADIENT,
                               1, 90, param1=50, param2=7, minRadius=23, maxRadius=55)
    circles = np.uint16(np.around(circles))
    circles = sorted(circles[0], key=(lambda x: x[2]))
    circles = np.uint16(np.around(circles))

    blues = np.zeros(img.shape, np.uint8)
    greens = np.zeros(img.shape, np.uint8)
    reds = np.zeros(img.shape, np.uint8)

    for i in circles[:]:
        cv2.circle(blues, (i[0], i[1]), i[2], np.random.randint(50, 255), -1)
        cv2.circle(greens, (i[0], i[1]), i[2], np.random.randint(50, 255), -1)
        cv2.circle(reds, (i[0], i[1]), i[2], np.random.randint(50, 255), -1)

    res = cv2.merge((blues, greens, reds))

    for i in range(len(circles)):
        circle = circles[i]
        cv2.putText(res, str(9-i), (circle[0]-10, circle[1]+10), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0))
    cv2.imwrite("5res_coins.jpg", res)

sort_coins()

#6
def split_coins():
    img = cv2.imread("coins_2.jpg")
    h, w, _ = img.shape

    def fill():
        for i in range(h):
            for j in range(w):
                if thresh[i, j] == 0:
                    new_img[i, j] = 255

    new_img = np.copy(img)
    img_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    _, thresh_1 = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    _, thresh_2 = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
    _, thresh_3 = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY)
    thresh = 255 - cv2.absdiff(thresh_2, thresh_1)
    thresh = cv2.medianBlur(thresh, 7)
    thresh = cv2.erode(thresh, np.ones((9, 9), np.uint8))
    fill()
    thresh = 255 - cv2.absdiff(thresh_3, thresh_1)
    thresh = cv2.medianBlur(thresh, 7)
    fill()

    img_diff = cv2.cvtColor(255 - cv2.absdiff(new_img, img), cv2.COLOR_BGR2GRAY)
    img_diff = cv2.medianBlur(img_diff, 9)
    _, img_diff = cv2.threshold(img_diff, 240, 255, cv2.THRESH_BINARY)
    img_diff = cv2.dilate(img_diff, np.ones((5, 5), np.uint8))
    kernel = np.zeros((19, 19), np.uint8)
    cv2.circle(kernel, (10, 10), 9, 1, -1)
    thresh = cv2.erode(img_diff, kernel)
    fill()

    img_diff = cv2.cvtColor(255 - cv2.absdiff(new_img, img), cv2.COLOR_BGR2GRAY)
    img_diff = cv2.medianBlur(img_diff, 9)
    _, img_diff = cv2.threshold(img_diff, 240, 255, cv2.THRESH_BINARY)
    img_diff = cv2.erode(img_diff, np.ones((9, 9), np.uint8))
    thresh = img_diff
    fill()

    text = new_img
    coins = 255 - cv2.absdiff(new_img, img)
    cv2.imwrite("6res_coins.jpg", coins)
    cv2.imwrite("6res_text.jpg", text)
#split_coins()

#7, 8
def sort_noisy_coins(path):
    img = cv2.imread(path, 0)
    img = cv2.medianBlur(img, 7)

    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    _, img = cv2.threshold(img, 45, 255, cv2.THRESH_BINARY)

    kernel = np.zeros((7, 7), np.uint8)
    cv2.circle(kernel, (3, 3), 1, 255, -1)
    img = cv2.dilate(img, np.ones((3, 3), np.uint8))

    #circles = cv2.HoughCircles(255 - img, cv2.cv.CV_HOUGH_GRADIENT, 1, 90, param1=43, param2=6, minRadius=23, maxRadius=60)
    circles = cv2.HoughCircles(255 - img, cv2.cv.CV_HOUGH_GRADIENT, 1, 90, param1=43, param2=6, minRadius=23, maxRadius=60)
    circles = np.uint16(np.around(circles))
    circles = sorted(circles[0], key=(lambda xx: xx[2]))
    circles = np.uint16(np.around(circles))

    blues = np.zeros(img.shape, np.uint8)
    greens = np.zeros(img.shape, np.uint8)
    reds = np.zeros(img.shape, np.uint8)
    x, y, z, cur, dif, cnt, rad = 0, 0, 0, 0, 10, 0, 0
    ind = 0
    group = []
    md = [0]
    col = [0]
    for i in range(len(circles)):
        c = circles[i]
        if i == 0 or abs(c[2] - cur) > dif:
            ind += 1
            if i != 0:
                md.append(rad / cnt)
                col.append(cnt)
            cur = c[2]
            rad = 0
            cnt = 0
            x = np.random.randint(50, 255)
            y = np.random.randint(50, 255)
            z = np.random.randint(50, 255)
        cnt += 1
        rad += c[2]
        group.append(ind)
        cv2.circle(blues, (c[0], c[1]), c[2], x, -1)
        cv2.circle(greens, (c[0], c[1]), c[2], y, -1)
        cv2.circle(reds, (c[0], c[1]), c[2], z, -1)

    md.append(rad / cnt)
    col.append(cnt)
    print len(circles)
    print 'groups:'
    print ind
    print md[1:]
    print 'amount in each group:'
    print col[1:]
    res = cv2.merge((blues, greens, reds))

    for i in range(len(circles)):
        circle = circles[i]
        cv2.putText(res, str(md[group[i]]), (circle[0]-15, circle[1]+10), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0))
    cv2.imwrite("7res_" + path, res)

#sort_noisy_coins("coins_noize_1.jpg")
#sort_noisy_coins("coins_noize_2.jpg")
#sort_noisy_coins("coins_noize_3.jpg")


