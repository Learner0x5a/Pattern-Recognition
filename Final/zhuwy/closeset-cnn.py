
import numpy as np 
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical,plot_model
from keras import backend as K
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
full_X = np.concatenate((GM_X,O_X),axis=0)
full_Y = np.concatenate((GM_Y,O_Y),axis=0)
full_Y = to_categorical(full_Y)
X_train, X_test, Y_train, Y_test = train_test_split(full_X,full_Y,test_size=0.3)


def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
 
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)

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
model.add(Dense(3,activation='softmax'))

adam = Adam()
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy",precision, recall, fmeasure])

model.fit(X_train, Y_train, batch_size=32, epochs=30,validation_data=(X_test,Y_test))

'''
epoch 30 
loss: 0.0682 - accuracy: 0.9704 - precision: 0.9710 - recall: 0.9710 - fmeasure: 0.9710 
- val_loss: 4.6051 - val_accuracy: 0.5968 - val_precision: 0.5969 - val_recall: 0.5969 - val_fmeasure: 0.5969
'''