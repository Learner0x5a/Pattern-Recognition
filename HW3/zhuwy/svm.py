import numpy as np
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

W1 = np.array([(1,1),(2,2),(3,3),(1,3),(3,1)])
W2 = np.array([(1,2),(2,1),(2,3),(3,2)])
W = np.vstack((W1,W2))
'''
SVM
'''
from sklearn.linear_model import LinearRegression as LR
labels = np.array([0,0,0,0,0,1,1,1,1])
def plot_hyperplane(model,X,y,h=0.01,draw_sv=False):
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('x')
    plt.ylabel('y')

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) # SVM的分割超平面
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='tab10', alpha=0.5)

    markers = ['o', 's', '^']
    colors = ['b', 'r', 'c']
    labels = np.unique(y)
    for label in labels:
        plt.scatter(X[y==label][:, 0], X[y==label][:, 1], c=colors[label], marker=markers[label])
    # 支持向量
    if draw_sv:
        sv = model.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')
    plt.savefig("svm.png")

from sklearn import svm
W = np.vstack((W1,W2))
model = svm.SVC(C=100.0)
model.fit(W,labels)
preds = model.predict(W)
print(mean_squared_error(preds,labels))
plot_hyperplane(model, W, labels)
