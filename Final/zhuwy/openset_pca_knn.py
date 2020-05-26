

# 用GLASS和MIRROR进行训练
# 用OTHERS进行测试
# 给出拒识方式

import pickle
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
np.random.seed(1)
import matplotlib
matplotlib.use('AGG')#或者PDF, SVG或PS
import matplotlib.pyplot as plt

f = open('RESIZE_X.pkl','rb')
X = pickle.load(f)
f.close()
GLASS,MIRROR,OTHERS = X
print(GLASS.shape, MIRROR.shape, OTHERS.shape)

GM_X = np.concatenate((GLASS,MIRROR),axis=0)
GM_Y = np.concatenate((np.zeros(GLASS.shape[0]),np.ones(MIRROR.shape[0])),axis=0)
print(GM_X.shape, GM_Y.shape)

N,x,y,z = GM_X.shape
GM_X = np.reshape(GM_X,(N,x*y*z))

O_X = OTHERS # 负样本
O_Y = np.full(O_X.shape[0],2) # 标签
N,x,y,z = O_X.shape
O_X = np.reshape(O_X,(N,x*y*z))

# PCA
pca = PCA(n_components=9)
newGM_X = pca.fit_transform(GM_X)
# 标准化
scaler = StandardScaler().fit(newGM_X)
newGM_X = scaler.transform(newGM_X)

# PCA
new_O_X = pca.transform(O_X)
# 标准化
new_O_X = scaler.transform(new_O_X)

# recover_GM_X = pca.inverse_transform(newGM_X)
# err = mean_squared_error(GM_X,recover_GM_X)
# print(pca.explained_variance_,err,newGM_X.shape,recover_GM_X.shape)

X_train, X_test, y_train, y_test = train_test_split(newGM_X,GM_Y,test_size=0.3)
print(y_test.shape,O_Y.shape)
X_test = np.concatenate((X_test,new_O_X[0:100]),axis=0)
y_test = np.concatenate((y_test,O_Y[0:100]),axis=0) # 负样本不参与训练，直接加入测试集
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train,y_train)
pred = model.predict(X_test)

# 拒识之前：mse: 1.1006493506493507 precision: 0.32588262697993 recall: 0.4837662337662338 f1: 0.38924413491531606
# 拒识之前，UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
# label 2 从来没有被预测到，所以F-score没有计算这项 label， 因此这种情况下 F-score 就被当作为 0.0 ；
# 根据距离拒识
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(X_train)
distances,indices = neigh.kneighbors(X_test,return_distance=True)
print(distances.shape,indices.shape)
print(max(distances),min(distances),np.mean(distances))
# print(np.where(distances<1)[0])
log = []
thres = np.linspace(1e-2,2)
for thre in thres:
    pred = model.predict(X_test)
    pred[np.where(distances<thre)[0]] = 2
    mse = mean_squared_error(y_test,pred)
    acc = accuracy_score(y_test,pred)
    recall = recall_score(y_test,pred, average='weighted')
    precision = precision_score(y_test,pred, average='weighted')
    f1 = f1_score(y_test,pred, average='weighted')
    # conf_mat = confusion_matrix(y_test,pred)
    print('thre:',thre,'mse:',mse,'precision:',precision,'recall:',recall,'f1:',f1)
    log.append((thre,mse,precision,recall,f1))
log = np.asarray(log)

plt.plot(log[:,0],log[:,1],c='r')
plt.xlabel('thre')
plt.ylabel('mse')
plt.savefig('PCA-thre-mse.png')
plt.clf()

plt.plot(log[:,0],log[:,2],c='r')
plt.xlabel('thre')
plt.ylabel('Precision')
plt.savefig('PCA-thre-Precision.png')
plt.clf()

plt.plot(log[:,0],log[:,3],c='r')
plt.xlabel('thre')
plt.ylabel('Recall')
plt.savefig('PCA-thre-Recall.png')
plt.clf()

plt.plot(log[:,0],log[:,4],c='r')
plt.xlabel('thre')
plt.ylabel('F1')
plt.savefig('PCA-thre-F1.png')
plt.clf()