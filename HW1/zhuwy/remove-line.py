# import cv2
# import numpy as np 
# from scipy.signal import convolve2d
# '''
# 我想在卷积的过程中加个判断，当前卷积区域内有无连通左右的线
# 分割依据：和最大（最白）的列，和最大（最白）的行，先按行分，再按列分
# '''
# # def process(img):
# #     '''
# #     smoothing
# #     '''
# #     img = cv2.medianBlur(img, 3) # 中值滤波


# #     img = img.astype(np.uint8)
# #     # print(img)
# #     img = img*255

# #     # img.save('0.png')
# #     return img

# # image_path = 'train/0.bmp'
# image_path = 'test/cap.bmp'
# img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
# # img = cv2.imread(image_path)
# # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #彩色转灰度
# # img = cv2.fastNlMeansDenoising(img,h=13)

# # img = cv2.GaussianBlur(img,(2,2),0)
# # img = cv2.medianBlur(img,2)
# # img = cv2.bilateralFilter(img,3,99,255)
# # img = cv2.GaussianBlur(img,(3,3),127)
# # img = cv2.blur(img,(2,2))
# img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 127, 7)

# def rm_line(img):
#     # H型滤波
#     a = np.full((45,),-1)
#     np.insert(a,0,[1,1])
#     np.append(a,[1,1])
#     b = np.full((45,),1)
#     np.insert(b,0,[1,1])
#     np.append(b,[1,1])
#     kernel = np.vstack((a,b,b,b,a))
#     lines = cv2.filter2D(img, -1, kernel=kernel)
#     # T型滤波
#     a = np.full((50,),-1)
#     np.insert(a,0,[1,1])
#     b = np.full((50,),1)
#     np.insert(b,0,[1,1])
#     kernel = np.vstack((a,b,b,b,a))
#     lines = cv2.filter2D(img, -1, kernel=kernel)

#     a = np.full((40,),-1)
#     np.append(a,[1,1])
#     b = np.full((40,),1)
#     np.append(b,[1,1])
#     kernel = np.vstack((a,b,b,b,a))
#     lines = cv2.filter2D(img, -1, kernel=kernel)





#     # lines = cv2.GaussianBlur(img,(55,3),0)
#     # lines = cv2.GaussianBlur(lines,(31,3),0)
#     # lines = cv2.GaussianBlur(lines,(15,3),0)
#     # lines = cv2.GaussianBlur(lines,(7,3),0)
#     lines = cv2.adaptiveThreshold(lines, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 7)
#     # lines =  cv2.medianBlur(lines,3)
#     img = 255 - lines + img
#     # img = lines
#     img =  cv2.medianBlur(img,5)
#     # img =  cv2.medianBlur(img,3)
#     # img =  cv2.medianBlur(img,3)
#     # img = cv2.blur(img,(2,2))
#     return img

# img = rm_line(img)

# # lines = cv2.GaussianBlur(img,(63,3),0)
# # # lines = cv2.GaussianBlur(lines,(31,3),0)
# # # lines = cv2.GaussianBlur(lines,(15,3),0)
# # # lines = cv2.GaussianBlur(lines,(7,3),0)
# # lines = cv2.adaptiveThreshold(lines, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 9)
# # lines =  cv2.medianBlur(lines,3)
# # img = 255 - lines + img
# # img =  cv2.medianBlur(img,3)


# # img = cv2.GaussianBlur(img,(3,3),127)
# # kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
# # img = cv2.filter2D(img, -1, kernel=kernel)






# # cv2.namedWindow("image")
# # cv2.createTrackbar("d","image",0,255,lambda x: None)
# # cv2.createTrackbar("sigmaColor","image",0,255,lambda x: None)
# # cv2.createTrackbar("sigmaSpace","image",0,255,lambda x: None)
# # while(1):
# #     d = cv2.getTrackbarPos("d","image")
# #     sigmaColor = cv2.getTrackbarPos("sigmaColor","image")
# #     sigmaSpace = cv2.getTrackbarPos("sigmaSpace","image")
# #     out_img = cv2.bilateralFilter(img,d,sigmaColor,sigmaSpace)
# #     cv2.imshow("out",out_img)
# #     k = cv2.waitKey(1) & 0xFF
# #     if k ==27:
# #         break
# # cv2.destroyAllWindows()

# # img=cv2.bilateralFilter(img,5,5,5)

# print(img.shape,np.max(img),np.min(img),np.mean(img))

# # img = process(img)
# cv2.imwrite('img.bmp',img)



# # '''
# # 10段平均的分割线
# # 减少滑动窗口计算量
# # '''
# # nb_col = img.shape[1]
# # average_idx = 0
# # for j in range(10):
# #     img_seg = img[:,j*int(nb_col/10):(j+1)*int(nb_col/10)]

# #     s = 0
# #     idx = 0
# #     mid = int(len(img_seg)/2)
# #     for i in range(mid-10,mid+10):
# #         if np.sum(img_seg[i]) > s:
# #             idx = i
# #     average_idx += idx
# # idx = int(average_idx/10)
# # up_img = img[:idx]
# # down_img = img[idx:]

# # # up_img = process(up_img)
# # cv2.imwrite('up.png',up_img)

# # # down_img = process(down_img)
# # cv2.imwrite('down.png',down_img)

import cv2
import numpy as np 

img = cv2.imread('test/cap.bmp',cv2.IMREAD_GRAYSCALE)
min_ = np.min(img)
# mask = np.where(img>min_+7,img,np.full(np.shape(img),np.mean(img)))
mask = np.where(img<=min_+7,255,0)
print(np.shape(mask))

cv2.imwrite('mask.bmp',mask)
mask = cv2.imread('mask.bmp',cv2.IMREAD_GRAYSCALE)
repair = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
cv2.imwrite('repair.bmp',repair)
img = cv2.adaptiveThreshold(repair, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 211, 7)
img = cv2.medianBlur(img,5)
img = cv2.medianBlur(img,5)
cv2.imwrite('binary-repair.bmp',img)