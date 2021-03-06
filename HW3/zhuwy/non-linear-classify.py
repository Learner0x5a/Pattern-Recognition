'''
作图
    超平面分割
    3d数据分布
    tsne
'''
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
fig = plt.figure()

W1 = np.array([(1,1),(2,2),(3,3),(1,3),(3,1)])
W2 = np.array([(1,2),(2,1),(2,3),(3,2)])
W = np.vstack((W1,W2))
x = W[:,0]
y = W[:,1]
# print(W,x,y)
Z = np.power(np.power(x-2,2) - np.power(y-2,2),2)[:,np.newaxis]
print(W,Z)

W = np.concatenate((W,Z),axis=1)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
X = np.linspace(0,4,150)
Y = np.linspace(0,4,150)
X, Y = np.meshgrid(X, Y)
Z = np.full(X.shape,1/2)
ax.plot_surface(X,Y,Z,color='green',alpha=0.2)
ax.set_zlabel('z') 
ax.set_ylabel('y')
ax.set_xlabel('x')
ax.scatter(W[0:5,0], W[0:5,1], W[0:5,2],c='b',label='W1')
ax.scatter(W[5:,0], W[5:,1], W[5:,2],c='r',label='W2')
plt.legend()
plt.savefig("3d.png")
plt.clf()

plt.scatter(W1[:,0],W1[:,1],marker='x',label='W1',c='b')
plt.scatter(W2[:,0],W2[:,1],marker='o',label='W2',c='r')
plt.xlim((0, 4))
plt.ylim((0, 4))
plt.xlabel('x')
plt.ylabel('y')

half = np.abs(X-2) - np.abs(Y-2)
X1 = X[np.where(half<=0)]
Y1 = 2 + np.sqrt(np.power(X1-2,2) + 1/2)
Y2 = 2 - np.sqrt(np.power(X1-2,2) + 1/2)
X2 = X[np.where(half<=0)]
Y3 = 2 + np.sqrt(np.power(X2-2,2) - 1/2)
Y4 = 2 - np.sqrt(np.power(X2-2,2) - 1/2)
plt.scatter(X1,Y1,s=1,c='g',label='hyperplane')
plt.scatter(X1,Y2,s=1,c='g')
plt.scatter(X2,Y3,s=1,c='g')
plt.scatter(X2,Y4,s=1,c='g')
plt.legend()
plt.savefig('2d.png')
plt.clf()



from sklearn.manifold import TSNE
def plot_with_labels(low_dim_embs, labels, filename):   # 绘制词向量图
    plt.figure()  
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        if label == 'W1':
            color = 'b'
        else:
            color = 'r'
        plt.scatter(x, y,c=color)	# 画点，对应low_dim_embs中每个词向量
        plt.xticks(()) # 不显示刻度
        plt.yticks(()) # 不显示刻度
        plt.xlabel('x')
        plt.ylabel('y')
        plt.annotate(label,	# 显示每个点对应哪个单词
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)

tsne = TSNE(n_components=2)
low_dim_embs = tsne.fit_transform(W)
labels = ['W1','W1','W1','W1','W1','W2','W2','W2','W2']
plot_with_labels(low_dim_embs, labels, 'tsne.png')
