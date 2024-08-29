import cv2
import numpy as np
from itertools import groupby
from itertools import chain
import statistics
import os

if not os.path.exists("Resources/Output_Line images/"):
    os.mkdir("Resources/Output_Line images/")
print("Word Segmenting Starts....")
for n in range(1, 21):
    file = "Resources/Line images/{}.tif".format(n)
    img = cv2.imread(file, 0)
    final_img = img.copy()
    row, col = img.shape

    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret1, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    th2 = th.copy()

    def counting(i, j, c):
        if c <= threshold_count and 0 < (i - 1) < row and 0 < (j - 1) < col and th2[i - 1][j - 1] == 0:
            th2[i - 1][j - 1] = 155
            c = c + 1
            i = i - 1
            j = j - 1
            c = counting(i, j, c)
            return c

        elif c <= threshold_count and 0 < (i - 1) < row and 0 < j < col and th2[i - 1][j] == 0:
            th2[i - 1][j] = 155
            c = c + 1
            i = i - 1
            j = j
            c = counting(i, j, c)
            return c

        elif c <= threshold_count and 0 < (i - 1) < row and 0 < (j + 1) < col and th2[i - 1][j + 1] == 0:
            th2[i - 1][j + 1] = 155
            c = c + 1
            i = i - 1
            j = j + 1
            c = counting(i, j, c)
            return c

        elif c <= threshold_count and 0 < i < row and 0 < (j - 1) < col and th2[i][j - 1] == 0:
            th2[i][j - 1] = 155
            c = c + 1
            i = i
            j = j - 1
            c = counting(i, j, c)
            return c

        elif c <= threshold_count and 0 < i < row and 0 < (j + 1) < col and th2[i][j + 1] == 0:
            th2[i][j + 1] = 155
            c = c + 1
            i = i
            j = j + 1
            c = counting(i, j, c)
            return c

        elif c <= threshold_count and 0 < (i + 1) < row and 0 < (j - 1) < col and th2[i + 1][j - 1] == 0:
            th2[i + 1][j - 1] = 155
            c = c + 1
            i = i + 1
            j = j - 1
            c = counting(i, j, c)
            return c

        elif c <= threshold_count and 0 < (i + 1) < row and 0 < j < col and th2[i + 1][j] == 0:
            th2[i + 1][j] = 155
            c = c + 1
            i = i + 1
            j = j
            c = counting(i, j, c)
            return c

        elif c <= threshold_count and 0 < (i + 1) < row and 0 < (j + 1) < col and th2[i + 1][j + 1] == 0:
            th2[i + 1][j + 1] = 155
            c = c + 1
            i = i + 1
            j = j + 1
            c = counting(i, j, c)
            return c

        else:
            return c


    c = 0
    lst = []
    threshold_count = 12
    for i in range(row):
        for j in range(col):
            if th2[i][j] == 0:
                c = counting(i, j, c)
                if c <= threshold_count:
                    lst.append(c)
            c = 0

    val = int(statistics.median(lst))
    if val % 2 == 0:
        val = val + 1

    th1 = cv2.GaussianBlur(th1, (val, val), 0)

    flag = 0
    for i in range(col):
        for j in range(row):
            if th[j][i] == 0:
                p, q = j, i
                flag = 1
                break
        if flag == 1:
            break
    for i in range(row):
        for j in range(q):
            if th[i][j] == 255:
                th[i][j] = 155

    flag = 0
    for i in range(col-1, 0, -1):
        for j in range(row-1, 0, -1):
            if th[j][i] == 0:
                a, b = j, i
                flag = 1
                break
        if flag == 1:
            break
    for i in range(row):
        for j in range(col-1, b, -1):
            if th[i][j] == 255:
                th[i][j] = 155

    ele = 255
    maxi = 0
    for i in range(row):
        res = [list(j) for i, j in groupby(th[i][q:b+1].tolist(), lambda x: x == ele) if not i]
        res1 = list(chain.from_iterable(res))
        cnt = res1.count(0)
        if cnt >= maxi:
            maxi = cnt
            s_r = i

    ele, lst = 0, []
    f = [list(j) for i, j in groupby(th[s_r][q:b+1].tolist(), lambda x: x == ele) if not i]

    for i in f:
        cnt_white = i.count(255)
        lst.append(cnt_white)
    median = int(statistics.median(lst))

    kernel = np.ones((median, median), np.uint8)
    img_dilation = cv2.dilate(th1, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    lst1, lst2 = [], []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        crop = final_img[y:y + h, x:x + w]
        lst1.append(x)
        lst2.append(crop)

    zipped_lists = zip(lst1, lst2)
    sorted_zipped_lists = sorted(zipped_lists)
    sorted_list1 = []
    for _, element in sorted_zipped_lists:
        sorted_list1.append(element)

    l = 1
    for crop_img in sorted_list1:
        p = "Resources/Output_Line images/Crop_{}/".format(n)
        if not os.path.exists(p):
            os.mkdir(p)
        cv2.imwrite(os.path.join(p, "{}.tif".format(l)), crop_img)
        l = l + 1

print("Words Successfully Segmented!!")