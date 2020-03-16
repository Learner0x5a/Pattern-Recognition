import numpy as np
import cv2 as cv

img = cv.imread('./test/huahen.bmp')
mask = cv.imread('./test/huahen_mask.bmp', 0)

dst = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)
cv.imshow('img', img)
cv.imshow('mask', mask)

cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('./test/7.bmp', dst)

img = cv.imread('./test/zaosheng.bmp')
dst = cv.medianBlur(img, 5)
# guass = cv.GaussianBlur(img, (3,3), 0)

cv.imshow('img', img)
cv.imshow('dst', dst)
# cv.imshow('guass', guass)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('./test/8.bmp', dst)