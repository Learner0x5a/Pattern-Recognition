import numpy as np 
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
fig = plt.figure()

f = open('testSet.txt','r')
lines = f.readlines()
f.close()
X = []
for line in lines:
    x = float(line.split()[0])
    y = float(line.split()[1])
    X.append((x,y))
X = np.asarray(X)


# # init_points = np.array([(-4.822,4.607),(-0.7188,-2.493),(4.377,4.864)])
# # init_points = np.array([(-3.594,2.857),(-0.6595,3.111),(3.998,2.519)])
# # init_points = np.array([(-0.7188,-2.493),(0.8458,-3.59),(1.149,3.345)])
# init_points = np.array([(-3.276,1.577),(3.275,2.958),(4.377,4.864)])
# model = KMeans(n_clusters=3,init=init_points,max_iter=300,verbose=1)

# init_points = np.array([(-0.00675,3.227),(-0.46,-2.77)])
# model = KMeans(n_clusters=2,init=init_points,max_iter=300,verbose=1)

init_points = np.array([(0.355,-3.36),(2.934,3.128),(-1.126,-2.302),(-2.947,3.236)])
model = KMeans(n_clusters=4,init=init_points,max_iter=300,verbose=1)

model.fit(X)
# print(model.cluster_centers_)
# init_points = model.cluster_centers_
pred = model.predict(X)
# print(pred)
plt.scatter(X[:,0],X[:,1],c=pred,alpha=0.5)
plt.scatter(init_points[:,0],init_points[:,1],c='r',alpha=0.5,label='initial points')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('kmeans.png')