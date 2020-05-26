
import numpy as np 
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical,plot_model
from keras.models import load_model
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
print(GM_X.shape, GM_Y.shape) # (N,144,144,3) (N,)
# GM_Y = to_categorical(GM_Y)

O_X = OTHERS # 负样本
O_Y = np.full(O_X.shape[0],2) # 标签

X_train, X_test, Y_train, Y_test = train_test_split(GM_X,GM_Y,test_size=0.3)
X_test = np.concatenate((X_test,O_X[0:100]),axis=0)
Y_test = np.concatenate((Y_test,O_Y[0:100]),axis=0) # 负样本不参与训练，直接加入测试集

# model = Sequential()
# model.add(BatchNormalization(input_shape=X_train.shape[1:]))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(BatchNormalization())
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(BatchNormalization())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))

# adam = Adam()
# model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

# model.fit(X_train, Y_train, batch_size=32, epochs=30)
# model.save('cnn.model')

np.set_printoptions(suppress = True)
from sklearn.metrics import mean_squared_error,accuracy_score,precision_score,recall_score,f1_score
model = load_model('cnn.model')

# thres = np.linspace(1e-2,0.5-1e-3)
thres = [0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.49,0.499,0.4999]
log = []
for thre in thres:
    pred = model.predict(X_test)
    glass = np.where(pred<0.5-thre)[0]
    mirror = np.where(pred>0.5+thre)[0]
    # others = np.where(0.5-thre<=pred<=0.5+thre)[0]
    # gm = np.concatenate((glass,mirror),axis=0)
    # others = np.delete(np.arange(pred.shape[0]),gm)
    others = np.where(np.abs(pred-0.5)<thre)[0]
    print(others)
    pred[glass] = 0
    pred[mirror] = 1
    pred[others] = 2
    
    
    mse = mean_squared_error(Y_test,pred)
    acc = accuracy_score(Y_test,pred)
    recall = recall_score(Y_test,pred, average='weighted')
    precision = precision_score(Y_test,pred, average='weighted')
    f1 = f1_score(Y_test,pred, average='weighted')
    print('thre:',thre,'mse:',mse,'precision:',precision,'recall:',recall,'f1:',f1)
    log.append((thre,mse,precision,recall,f1))
log = np.asarray(log)
print(thres)
plt.plot(log[:,0],log[:,1],c='r')
plt.xlabel('thre')
plt.ylabel('mse')
plt.savefig('CNN-thre-mse.png')
plt.clf()

plt.plot(log[:,0],log[:,2],c='r')
plt.xlabel('thre')
plt.ylabel('Precision')
plt.savefig('CNN-thre-Precision.png')
plt.clf()

plt.plot(log[:,0],log[:,3],c='r')
plt.xlabel('thre')
plt.ylabel('Recall')
plt.savefig('CNN-thre-Recall.png')
plt.clf()

plt.plot(log[:,0],log[:,4],c='r')
plt.xlabel('thre')
plt.ylabel('F1')
plt.savefig('CNN-thre-F1.png')
plt.clf()