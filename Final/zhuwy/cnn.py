
import numpy as np 
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical,plot_model
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

X_train, X_test, Y_train, Y_test = train_test_split(GM_X,GM_Y,test_size=0.3)


model = Sequential()
model.add(BatchNormalization(input_shape=X_train.shape[1:]))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

adam = Adam()
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])
plot_model(model,'cnn.png',show_shapes=True,rankdir='LR')
model.fit(X_train, Y_train, batch_size=32, epochs=30)
test_loss,test_acc = model.evaluate(X_test, Y_test, batch_size=32)
pred = model.predict(X_test)
zeros = np.where(pred<0.5)[0]
ones = np.where(pred>=0.5)[0]
pred[zeros] = 0
pred[ones] = 1
from sklearn.metrics import mean_squared_error,accuracy_score,precision_score,recall_score,f1_score
mse = mean_squared_error(Y_test,pred)
acc = accuracy_score(Y_test,pred)
recall = recall_score(Y_test,pred)
precision = precision_score(Y_test,pred)
f1 = f1_score(Y_test,pred)
print('mse:',mse,'precision:',precision,'recall:',recall,'f1:',f1)
