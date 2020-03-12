import cv2
import numpy as np 
img = cv2.imread('test/noise.bmp',cv2.IMREAD_GRAYSCALE)
img = cv2.GaussianBlur(img,(5,5),0)
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 8)
img = cv2.medianBlur(img,5)
img = cv2.medianBlur(img,3)

cv2.imwrite('denoise.bmp',img)