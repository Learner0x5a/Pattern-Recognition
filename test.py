import cv2
import numpy as np 

img = cv2.imread('denoise.bmp',cv2.IMREAD_GRAYSCALE)
img = np.array(img)

def R(f,g):
    l = max(max(f.shape),max(g.shape))
    f = np.pad(f,((int((l-f.shape[0])/2),int((l-f.shape[0])/2)),(int((l-f.shape[1])/2),int((l-f.shape[1])/2))),'constant',constant_values=(255,255))
    g = np.pad(g,((int((l-g.shape[0])/2),int((l-g.shape[0])/2)),(int((l-g.shape[1])/2),int((l-g.shape[1])/2))),'constant',constant_values=(255,255))

    l = max(max(f.shape),max(g.shape))
    f = np.pad(f,((0,l-f.shape[0]),(0,l-f.shape[1])),'constant',constant_values=(255,255))
    g = np.pad(g,((0,l-g.shape[0]),(0,l-g.shape[1])),'constant',constant_values=(255,255))
    cv2.imwrite('tmp/f.bmp',f)
    cv2.imwrite('tmp/g.bmp',g)
    r1 = np.sqrt(np.sum(np.power(f,2)))
    r2 = np.sqrt(np.sum(np.power(g,2)))
    conv = np.sum(np.multiply(f,g))
    res = conv/(r1*r2)
    return res



def split(img):
    nb_col = img.shape[1]
    average_idx = 0
    for j in range(10):
        img_seg = img[:,j*int(nb_col/10):(j+1)*int(nb_col/10)]

        s = 0
        idx = 0
        mid = int(len(img_seg)/2)
        for i in range(mid-10,mid+10):
            if np.sum(img_seg[i]) > s:
                idx = i
        average_idx += idx
    idx = int(average_idx/10)
    up_img = img[:idx]
    down_img = img[idx:]
    return up_img,down_img


# def bin(img):
#     nb_col = img.shape[1]
#     for j in range(10):
#         img_seg = img[:,j*int(nb_col/10):(j+1)*int(nb_col/10)]
#         thre, img_seg = cv2.threshold(img_seg,np.mean(img_seg),255,cv2.THRESH_BINARY)
#         img[:,j*int(nb_col/10):(j+1)*int(nb_col/10)] = img_seg
#     return img

# img = np.pad(img,((1,1),(1,1)),'constant',constant_values=(255,255)) # 补一圈边界
up,down=split(img)
up = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 127, 7)
down = cv2.adaptiveThreshold(down, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 127, 7)
up = cv2.medianBlur(up,3)
down = cv2.medianBlur(down,3)
cv2.imwrite('up.png',up)
cv2.imwrite('down.png',down)
# cv2.imwrite('test.bmp',img)

up_split = (np.sum(255-up,axis=0)/255 < 6).astype(np.uint8) # 列分割，1为黑色点少于6的列
down_split = (np.sum(255-down,axis=0)/255 < 6).astype(np.uint8) 
topredict = []

boarder=[]
for i in range(len(up_split)-1):# 分割边界
    if not up_split[i] == up_split[i+1]: 
        boarder.append(i)
    elif i == 0 or i == len(up_split)-2:
        boarder.append(i)
# boarder.append(up.shape[1])
i = 0
while i < len(boarder)-1:
    split_col = up[:,boarder[i]:boarder[i+1]]

    row_split = (np.sum(255-split_col,axis=1)/255 < 2).astype(np.uint8) # 行分割
    row_boarder = []
    for k in range(len(row_split)-1):# 分割边界
        if not row_split[k] == row_split[k+1]:
            row_boarder.append(k)
        elif k == 0 or k == len(row_split)-2:
            row_boarder.append(k)
    
    # j = 0
    # while j < len(row_boarder)-1:
    #     if abs(row_boarder[j+1] - row_boarder[j]) > int(up.shape[0]/2):
    #         split_col_row = split_col[row_boarder[j]:row_boarder[j+1],:]
    #         cv2.imwrite('tmp/up_'+str(i)+'.bmp',split_col_row)
    #         # print(np.shape(split_col_row))
    #         topredict.append(split_col_row)
    #     j += 1



    # cv2.imwrite('tmp/up_'+str(i)+'.bmp',split_col)
    # topredict.append(split_col)
    i += 2


# split = []
if not len(boarder) % 2 ==0:
    print('边界划分错误')



boarder=[]
for i in range(len(down_split)-1): # 合并分类边界
    if not down_split[i] == down_split[i+1]:
        boarder.append(i)
i = 0
while i < len(boarder)-1:
    split_col = down[:,boarder[i]:boarder[i+1]]

    row_split = (np.sum(255-split_col,axis=1)/255 < 2).astype(np.uint8) # 行分割
    row_boarder = []
    for k in range(len(row_split)-1):# 分割边界
        if not row_split[k] == row_split[k+1]:
            row_boarder.append(k)
        elif k == 0 or k == len(row_split)-2:
            row_boarder.append(k)
    j = 0
    while j < len(row_boarder)-1:
        if abs(row_boarder[j+1] - row_boarder[j]) > int(down.shape[0]/2):
            split_col_row = split_col[row_boarder[j]:row_boarder[j+1],:]
            cv2.imwrite('tmp/down_'+str(i)+'.bmp',split_col_row)
            # print(np.shape(split_col_row))
            topredict.append(split_col_row)
        j += 1

    # cv2.imwrite('tmp/down_'+str(i)+'.bmp',split_col)
    # topredict.append(split_col)
    i += 2
# split = [boarder[0]]
if not len(boarder) % 2 == 0:
    print('边界划分错误')




import os 

DICT = []
LABEL = []
for root,parent,files in os.walk('chars'):
    for f_img in files:
        img = cv2.imread(os.path.join(root,f_img),cv2.IMREAD_GRAYSCALE)
        label = f_img.split('.')[0][-1]
        DICT.append(img)
        LABEL.append(label)


for num in topredict:
    max_r = 0
    for char in DICT: 
        this_r = R(num,char)
        if this_r > max_r:
            max_r = this_r
            pred = LABEL[DICT.index(char)]    
    print(pred)



# for num in DICT:
#     (row,col) = np.shape(up)
#     (w,h) = np.shape(num)
#     label = LABEL[DICT.index(num)]
#     for i in range(row-h):
#         for j in range(col-w):
#             window = up[i:i+h,j:j+w]
#             # while(1):
#             #     cv2.imshow("out",window)
#             #     k = cv2.waitKey(1) & 0xFF
#             #     if k ==27:
#             #         break
            
# (row,col) = np.shape(up)
# (w,h) = (31,31) # sliding window
# for i in range(row-h):
#     for j in range(col-w):
#         window = up[i:i+h,j:j+w]
#         max_r = 0
#         pred = ''
#         for num in DICT: 
#             this_r = R(window,num)
#             if this_r > max_r:
#                 max_r = this_r
#                 pred = LABEL[DICT.index(num)]
#         if max_r>0.9:
#             print(max_r,pred)

#         # while(1):
#         #     cv2.imshow("out",window)
#         #     k = cv2.waitKey(1) & 0xFF
#         #     if k ==27:
#         #         break
        
