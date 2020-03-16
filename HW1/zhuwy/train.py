import os,cv2
import numpy as np 

out_dir = 'chars'
for root,parent,files in os.walk('train'):
    for f_img in files:
        img = cv2.imread(os.path.join(root,f_img),cv2.IMREAD_GRAYSCALE)
        thre, img = cv2.threshold(img,np.mean(img),255,cv2.THRESH_BINARY)
        # img = cv2.medianBlur(img,3)
        print(img/255)
        #print(np.shape(img))
        #cv2.imwrite(os.path.join(out_dir,(root+f_img).replace('/','_')),img)