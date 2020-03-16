import cv2
import numpy as np 
img = cv2.imread('test/1.bmp',cv2.IMREAD_GRAYSCALE)
base = np.shape(img)

img = cv2.imread('test/extra1.bmp',cv2.IMREAD_GRAYSCALE)
this_shape = np.shape(img)
#None是输出图像的尺寸大小，fx和fy是缩放因子
#cv2.INTER_CUBIC 是插值方法，一般默认为cv2.INTER_LINEAR
img = cv2.resize(img,None,fx=base[0]/this_shape[0],fy=base[1]/this_shape[1],interpolation=cv2.INTER_LINEAR)  
cv2.imwrite('resize-extra1.bmp',img)


img = cv2.imread('test/extra2.bmp',cv2.IMREAD_GRAYSCALE)
this_shape = np.shape(img)
img = cv2.resize(img,None,fx=base[0]/this_shape[0],fy=base[1]/this_shape[1],interpolation=cv2.INTER_CUBIC)  
cv2.imwrite('resize-extra2.bmp',img)