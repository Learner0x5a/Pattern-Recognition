from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import numpy as np 
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
fig = plt.figure()

x = np.arange(-6,6,0.1)
y = np.arange(-6,6,0.1)
Zx,Zy = np.meshgrid(x,y)
Zx = np.reshape(Zx,(Zx.shape[0]*Zx.shape[1],))
Zy = np.reshape(Zy,(Zy.shape[0]*Zy.shape[1],))
X_test = np.array(list(zip(Zx,Zy)))
print(X_test.shape)



f = open('trainData.txt','r')
plt_x1 = []
plt_y1 = []
plt_x2 = []
plt_y2 = []
label_train = []
for line in f.readlines():
    data = line.split()
    if data[2] == '1':
        plt_x1.append(float(data[0]))
        plt_y1.append(float(data[1]))
    else:
        plt_x2.append(float(data[0]))
        plt_y2.append(float(data[1]))
    label_train.append(float(data[2])-1)
f.close()

x_train = plt_x1 + plt_x2
y_train = plt_y1 + plt_y2
x_train = np.array(x_train)[:,np.newaxis]
y_train = np.array(y_train)[:,np.newaxis]

X = np.concatenate((x_train,y_train),axis=1)
label_train = np.array(label_train)

print(X.shape,label_train.shape)


for k in range(1,10):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X,label_train)

    label_test = model.predict(X_test)

    plt.scatter(X_test[:,0],X_test[:,1],c=label_test)
    plt.scatter(plt_x1,plt_y1,c='r')
    plt.scatter(plt_x2,plt_y2,c='g')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('k='+str(k))
    plt.savefig('images/knn-'+str(k)+'.png')
    plt.clf()

