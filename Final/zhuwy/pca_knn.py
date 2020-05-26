import pickle
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

log = []
for k in range(1,12):
    pca = PCA(n_components=k)
    newGM_X = pca.fit_transform(GM_X)
    scaler = StandardScaler().fit(newGM_X)
    newGM_X = scaler.transform(newGM_X)
    # recover_GM_X = pca.inverse_transform(newGM_X)
    # err = mean_squared_error(GM_X,recover_GM_X)
    # print(pca.explained_variance_,err,newGM_X.shape,recover_GM_X.shape)

    X_train, X_test, y_train, y_test = train_test_split(newGM_X,GM_Y,test_size=0.3)
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test,pred)
    acc = accuracy_score(y_test,pred)
    recall = recall_score(y_test,pred)
    precision = precision_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    conf_mat = confusion_matrix(y_test,pred)
    print(k,'mse:',mse,'precision:',precision,'recall:',recall,'f1:',f1)
    log.append((k,mse,precision,recall,f1,conf_mat))

log = np.asarray(log)
np.save('log.npy',log)


log = np.load('log.npy',allow_pickle=True)

plt.plot(log[:,0],log[:,1],c='r')
plt.xlabel('k')
plt.ylabel('mse')
plt.savefig('PCA-k-mse.png')
plt.clf()

plt.plot(log[:,0],log[:,2],c='r')
plt.xlabel('k')
plt.ylabel('Precision')
plt.savefig('PCA-k-Precision.png')
plt.clf()

plt.plot(log[:,0],log[:,3],c='r')
plt.xlabel('k')
plt.ylabel('Recall')
plt.savefig('PCA-k-Recall.png')
plt.clf()

plt.plot(log[:,0],log[:,4],c='r')
plt.xlabel('k')
plt.ylabel('F1')
plt.savefig('PCA-k-F1.png')
plt.clf()