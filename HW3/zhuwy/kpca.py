import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('AGG')#或者PDF, SVG或PS
import matplotlib.pyplot as plt

W1 = np.array([(1,1),(2,2),(3,3),(1,3),(3,1)])
W2 = np.array([(1,2),(2,1),(2,3),(3,2)])
W = np.vstack((W1,W2))
# x1 = W1[:,0]
# y1 = W1[:,1]
# x2 = W2[:,0]
# y2 = W2[:,1]
# print(W,x,y)

from sklearn.linear_model import LinearRegression as LR
labels = np.array([0,0,0,0,0,1,1,1,1])
# # for k in range(3,6):
# model = KernelPCA(n_components=3,kernel='rbf') # 2dim -> kdim
# W_kpca = model.fit_transform(W)
# # print(W_kpca)
def plot_hyperplane(clf, X, y, 
                    h=0.02, 
                    draw_sv=False, 
                    title='hyperplan'):
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # SVM的分割超平面
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='hot', alpha=0.5)

    markers = ['o', 's', '^']
    colors = ['b', 'r', 'c']
    labels = np.unique(y)
    for label in labels:
        plt.scatter(X[y==label][:, 0], 
                    X[y==label][:, 1], 
                    c=colors[label], 
                    marker=markers[label])
    # 画出支持向量
    if draw_sv:
        sv = clf.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')
    plt.savefig("kpca.png")

from sklearn import svm
model = svm.SVC(C=100.0)
model.fit(W,labels)
preds = model.predict(W)
print(mean_squared_error(preds,labels))
# plt.scatter(W[0:5,0],W[0:5,1],c='black',marker='x',label='W1')
# plt.scatter(W[5:,0],W[5:,1],c='red',marker='o',label='W2')
# plt.legend()
# plt.savefig("kpca.png")
plot_hyperplane(model,W,labels, h=0.01, 
                title='Maximum Margin Hyperplan')

# # from sklearn.manifold import TSNE

# # def plot_with_labels(low_dim_embs, labels, filename):   # 绘制词向量图
# #     assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
# #     print('Start drawing......')
# #     plt.figure()  
# #     for i, label in enumerate(labels):
# #         x, y = low_dim_embs[i, :]
# #         plt.scatter(x, y)	# 画点，对应low_dim_embs中每个词向量
# #         plt.annotate(label,	# 显示每个点对应哪个单词
# #                      xy=(x, y),
# #                      xytext=(5, 2),
# #                      textcoords='offset points',
# #                      ha='right',
# #                      va='bottom')
# #     plt.savefig(filename)
# #     print('Done!')
# #     # plt.show()
# # # print(W_kpca)
# # tsne = TSNE(n_components=2)
# # low_dim_embs = tsne.fit_transform(W_kpca)
# # labels = ['0','0','0','0','0','1','1','1','1']
# # plot_with_labels(low_dim_embs, labels, 'tsne.png')

# # from sklearn.externals import joblib #jbolib模块

# # #保存Model(注:save文件夹要预先建立，否则会报错)
# # joblib.dump(model, 'model.pkl')

# # x1 = W_kpca[0:5,0]
# # y1 = W_kpca[0:5,1]
# # x2 = W_kpca[5:,0]
# # y2 = W_kpca[5:,1]
# # plt.xlabel(u'x',FontSize=16)
# # plt.ylabel(u'y',FontSize=16)
# # # plt.title(img,fontsize='large',fontweight='bold')
# # plt.scatter(x1,y1,c='black',marker='x',label='W1')
# # plt.scatter(x2,y2,c='red',marker='o',label='W2')
# # plt.legend()
# # plt.savefig("kpca.png")

# # x1 = W_kpca[0:5,0]
# # y1 = W_kpca[0:5,1]
# # z1 = W_kpca[0:5,2]
# # x2 = W_kpca[5:,0]
# # y2 = W_kpca[5:,1]
# # z2 = W_kpca[5:,2]

# # from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
# # fig = plt.figure()
# # ax = Axes3D(fig)
# # ax.scatter(x1, y1, z1,c='b')
# # ax.scatter(x2, y2, z2,c='r')
# # plt.savefig("kpca.png")


