import cv2,os,json
import numpy as np 
import pickle

# def myround(x):
#     int_x = int(x)
#     if x - int_x >= 0.5:
#         return int_x + 1
#     else:
#         return int_x

# def swap(x,y):
#     tmp = x 
#     x = y 
#     y = tmp 
#     return x,y

# def preprocess(x1,x2,y1,y2):
#     if x1 < 0:
#         x1 = 0
#     if x2 < 0:
#         x2 = 0
#     if y1 < 0:
#         y1 = 0
#     if y2 < 0:
#         y2 = 0
    
#     x1 = myround(x1)
#     x2 = myround(x2)
#     if x1 > x2:
#         x1,x2 = swap(x1,x2)
#     y1 = myround(y1)
#     y2 = myround(y2)
#     if y1 > y2:
#         y1,y2 = swap(y1,y2)
#     return x1,x2,y1,y2

# GLASS = []
# MIRROR = []
# OTHERS = []


# # labeldict = {'glass':0,'mirror':1,'others':2}
# def dataloader(workdir):
#     # 读取坐标、标签
#     for root,parent,files in os.walk(workdir+'/label'):
#         for file_ in files:
#             f = open(os.path.join(root,file_),'r')
#             load = json.load(f)
#             f.close()
#             count = 0
#             for sample in load['shapes']:
#                 # label = labeldict[sample['label']]
#                 label = sample['label']
#                 [[y1,x1],[y2,x2]] = sample['points']
#                 imgf = file_.split('.')[0]
#                 x1,x2,y1,y2 = preprocess(x1,x2,y1,y2)
#                 print(file_,imgf)
#                 # 裁剪图片
#                 img = cv2.imread(os.path.join(workdir,'rgb',imgf+'.jpg'),cv2.IMREAD_COLOR)
#                 # print(img.shape)
#                 crop_img = img[x1:x2,y1:y2]
#                 print(x1,x2,y1,y2,crop_img.shape)
#                 # cv2.imwrite(workdir+'-crop/'+label+'_'+imgf+'_'+str(count)+'.jpg',crop_img)
#                 if label == 'glass':
#                     GLASS.append(crop_img)
#                 elif label == 'mirror':
#                     MIRROR.append(crop_img)
#                 else:
#                     OTHERS.append(crop_img)
#                 count += 1 

# dataloader('train')
# dataloader('test')
# GLASS = np.array(GLASS)
# MIRROR = np.array(MIRROR)
# OTHERS = np.array(OTHERS)
# print(GLASS.shape, MIRROR.shape, OTHERS.shape) # 380 310 562
# # X = {'glass':GLASS,'mirror':MIRROR,'others':OTHERS}
# X = [GLASS,MIRROR,OTHERS]
# f = open('X.pkl','wb')
# pickle.dump(X,f)
# f.close()

f = open('X.pkl','rb')
X = pickle.load(f)
f.close()
GLASS,MIRROR,OTHERS = X

def avg_shape(X):
    total_x = 0
    total_y = 0
    count = 0
    for DATA in X:
        for img in DATA:
            x,y,z = img.shape
            total_x += x
            total_y += y
            count += 1
    avg_x = total_x//count
    avg_y = total_y//count
    print(avg_x,avg_y)
    return avg_x,avg_y

# x,y = avg_shape(X) # 140 132
x,y = 144,144

# count =  0
# for DATA in X:
#     for img in DATA:
#         newimg = cv2.resize(img,(x,y),interpolation=cv2.INTER_LINEAR)
#         # cv2.imwrite('resize/'+str(count)+'.jpg',newimg)
#         print(newimg.shape)
#         count += 1

def resize(DATA):
    NEWDATA = []
    for img in DATA:
        newimg = cv2.resize(img,(x,y),interpolation=cv2.INTER_LINEAR)
        NEWDATA.append(newimg)
    return np.array(NEWDATA)

GLASS = resize(GLASS)
MIRROR = resize(MIRROR)
OTHERS = resize(OTHERS)
print(GLASS.shape, MIRROR.shape, OTHERS.shape) # (383, 144, 144, 3) (310, 144, 144, 3) (562, 144, 144, 3)
X = [GLASS,MIRROR,OTHERS]
f = open('RESIZE_X.pkl','wb')
pickle.dump(X,f)
f.close()