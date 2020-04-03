from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import numpy as np 
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

plt.scatter(X_test[:,0],X_test[:,1],s=1)

plt.savefig('knn.png')

model = KNeighborsClassifier(n_neighbors=3)

########### ??? 标签是什么

# model.fit(X)  