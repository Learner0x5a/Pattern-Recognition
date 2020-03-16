# encoding:utf-8
# Adapted from this blog[https://www.cnblogs.com/FHC1994/p/9123393.html] 
# classify several numbers per time
import cv2 as cv
import numpy as np
import os

def local_threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)
    return binary

def template_demo(tp_path, target_path):
    tp0 = cv.imread(tp_path)
    target = cv.imread(target_path)
    # tp0 = local_threshold(tp0)
    # target = local_threshold(target)
    # cv.namedWindow('template image', cv.WINDOW_NORMAL)
    # cv.imshow('template image', tp0)
    # cv.namedWindow('target image', cv.WINDOW_NORMAL)
    # cv.imshow('target image', target)
    methods = [cv.TM_SQDIFF_NORMED]
    # methods = [cv.TM_SQDIFF_NORMED]
    th, tw = tp0.shape[:2]
    for md in methods:
        print(md)
        result = cv.matchTemplate(target, tp0, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        # print('-=-=-=-=-=-')
        # print(max_val)
        # print(min_val)
        # print(result.shape)
        # print('=-=-=-=-=-=')
        threshold = 0.013
        loc = np.where(result <= threshold)
        # print(loc)
        values = [(0, 0, 1)]
        for t1 in zip(*loc[::-1]):
            # print(t1)
            va = result[t1[1], t1[0]]
            flag = 1
            for num in range(len(values)):
                item = values[num]
                if (t1[0]-item[0])**2 + (t1[1]-item[1])**2 < 100:
                    flag = 0
                    if va < item[2]:
                        values[num] = (t1[0], t1[1], va)
                    break
            if flag == 1:
                values.append(((t1[0], t1[1], va)))
        for t1 in values:
            if t1[2] == 1:
                continue
            cv.rectangle(target, (t1[0], t1[1]), (t1[0]+tw, t1[1]+th), (0, 0, 255), 2)
        cv.namedWindow('match-'+np.str(md), cv.WINDOW_NORMAL)
        cv.imshow('match-'+np.str(md), target)


if __name__ == "__main__":

    target_path = './test/'
    train_path = './train/'
    for i in range(8):
        target = target_path + str(i+1) + '.bmp'
        for (root, dirs, files) in os.walk('train'):
            for tp in files:
                print(train_path+tp)
                print(target)
                template_demo(train_path+tp, target)
                cv.waitKey(0)
    cv.destroyAllWindows()
