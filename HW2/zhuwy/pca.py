import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('AGG')#或者PDF, SVG或PS
import matplotlib.pyplot as plt

def myPCA(img):
    X = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    X = np.array(X)
    # print(X.shape)
    max_k = X.shape[1]

    plt_x = []
    plt_y = []
    for k in range(1,max_k):
        pca = PCA(n_components=k)
        newX = pca.fit_transform(X)
        recover_X = pca.inverse_transform(newX)
        err = mean_squared_error(X,recover_X)
        # print('k = ',str(k),' Error:',err)
        plt_x.append(k)
        plt_y.append(err)

    print(img,'-> k = ',str(k),' Error:',err,' Feature weights:',pca.explained_variance_ratio_[0:3])


    # plt.scatter(plt_x,plt_y,c='black',marker='s',s=8,alpha=0.5)
    plt.xlabel(u'k',FontSize=16)
    plt.ylabel(u'MSE',FontSize=16)
    plt.title(img,fontsize='large',fontweight='bold')
    plt.plot(plt_x,plt_y,color='r')

    plt.savefig("k-err"+img+".png")
    plt.clf()

myPCA('face.jpg')
myPCA('face1.jpg')
myPCA('noface.jpg')