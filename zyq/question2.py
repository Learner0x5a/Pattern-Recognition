import cv2
import numpy as np
import os


def CalBlackNum(img):
    (h,w) = img.shape
    num = []
    area = [(0,0),(0,w//2),(h//4,0),(h//4,w//2),(h//2,0),(h//2,w//2),(h//4*3,0),(h//4*3,w//2)]
    for item in area:
        
        for i in range(h//4):
            la = 0
            for j in range(w//2):
                if img[i+item[0],j+item[1]]==0:
                    la+=1
            num.append(la)
    return num
'''水平投影'''
def getHProjection(image):
    hProjection = np.zeros(image.shape,np.uint8)
    #图像高与宽
    (h,w)=image.shape
    #长度与图像高度一致的数组
    h_ = [0]*h
    #循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y,x] == 255:
                h_[y]+=1
    print(h_)
    print(max(h_))
    print((h,w))
    #la = max(h_)
    #lala = [i-la+255 for i in h_]
    #h_ = lala
    #绘制水平投影图像
    for y in range(h):
        for x in range(h_[y]):
            hProjection[y,x] = 255

    cv2.imshow('hProjection2',hProjection)
    
    cv2.waitKey(0)
    return h_

# 局部阈值
def local_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
    # 自适应阈值化能够根据图像不同区域亮度分布，改变阈值
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
    cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
    cv2.imshow("binary", binary)
    cv2.waitKey(0)
    return binary

def getVProjection(image,i,path):
    vProjection = np.zeros(image.shape,np.uint8)
    #图像高与宽
    (h,w) = image.shape
    #长度与图像宽度一致的数组
    w_ = [0]*w
    #循环统计每一列白色像素的个数
    for x in range(w):
        for y in range(h):
            if image[y,x] == 255:
                w_[x]+=1
    print('++++++')
    print(w_)
    print(max(w_))
    print((h,w))    
    #绘制垂直平投影图像
    for x in range(w):
        for y in range(h-w_[x],h):
            vProjection[y,x] = 255
    cv2.imshow('vProjection',vProjection)
    #cv2.imwrite(path+str(i)+'.bmp',vProjection)
    cv2.waitKey(0)
    return w_
 
if __name__ == "__main__":
    #读入原始图像
    origineImage = cv2.imread('./test/3.bmp')

    img = local_threshold(origineImage)
    # cv2.imshow('binary',img)
    #图像高与宽
    (h,w)=img.shape
    Position = []
    #水平投影
    H = getHProjection(img)
 
    start = 0
    H_Start = []
    H_End = []
    #根据水平投影获取垂直分割位置
    for i in range(len(H)):
        if H[i] < w and start ==0:
            H_Start.append(i)
            start = 1
        if H[i] >= w and start == 1:
            H_End.append(i)
            start = 0
    #分割行，分割之后再进行列分割并保存分割位置
    H_End.append(len(H)-1)
    print('------')
    print(len(H))
    print(H_Start)
    print(H_End)
    count = 0
    abc = []

    for i in range(len(H_Start)):
        #获取行图像
        #print([H_Start[i],H_End[i]])
        #print([0,w])
        #print(img.shape)
        cropImg = img[H_Start[i]:H_End[i], 0:w]
        #cropImg = local_threshold(cropImg)
        (h1,w1)=cropImg.shape
        #cv2.imshow('test',cropImg)
        cv2.waitKey(0)
        #cv2.imshow('cropImg',cropImg)
        #对行图像进行垂直投影
        W = getVProjection(cropImg,i,'test_projection/')
        Wstart = 0
        Wend = 0
        W_Start = 0
        W_End = 0
        for j in range(len(W)):
            if W[j] < h1 and Wstart ==0:
                W_Start =j
                Wstart = 1
                Wend=0
            if W[j] >= h1 and Wstart == 1:
                W_End =j
                Wstart = 0
                Wend=1
            if Wend == 1:
                Position.append([W_Start,H_Start[i],W_End,H_End[i]])
                print([W_Start,H_Start[i],W_End,H_End[i]])
                op = img[H_Start[i]:H_End[i],W_Start:W_End]
                op = cv2.resize(op, (24, 24))
                cv2.imwrite('./test_projection/'+str(count)+".bmp",op)
                tem = CalBlackNum(op)
                abc.append(tem)
                count+=1
                #cv2.waitKey(0)
                Wend =0
    #根据确定的位置分割字符
    for m in range(len(Position)):
        cv2.rectangle(origineImage, (Position[m][0],Position[m][1]), (Position[m][2],Position[m][3]), (0 ,229 ,238), 1)
    cv2.imshow('image',origineImage)
    cv2.waitKey(0)
    print('--------------------------------------------------------')
    for i in abc:
        print(i)
    templates = []
    print('--------------------------------------------------------')
    for (root, dirs, files) in os.walk('train'):
        for tp in files:
            path = './train/'+ tp
            op = cv2.imread(path)
            op = local_threshold(op)
            op = cv2.resize(op, (24, 24))
            tem = CalBlackNum(op)
            templates.append(tem)
            print(tem)
    values = []
    abc = np.array(abc)
    templates = np.array(templates)
    for i in abc:
        yui = []
        for j in templates:
            cen1 = i-np.mean(i)
            cen2 = j-np.mean(j)
            nor1 = cen1/np.std(cen1)
            nor2 = cen2/np.std(cen2)
            op = np.linalg.norm(nor1-nor2)
            yui.append(op)
        n = np.argmin(yui)
        values.append(n)
    print(values)