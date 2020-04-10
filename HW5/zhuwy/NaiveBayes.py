from math import log # natual log
import numpy as np 
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
fig = plt.figure()

priors = [0.5,0.5]
X1 = np.array([(3,4),(3,8),(2,6),(4,6)])
X2 = np.array([(3,0),(3,-4),(1,-2),(5,-2)])
X = np.concatenate((X1,X2),axis=0)
Y = np.concatenate((np.zeros(X1.shape[0]),np.ones(X2.shape[0])),axis=0)
plt.scatter(X1[:,0],X1[:,1],label='X1')
plt.scatter(X2[:,0],X2[:,1],label='X2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
print(X1,X2)

m1 = np.mean(X1,axis=0)
m2 = np.mean(X2,axis=0)
m1_col = m1[:,np.newaxis]
m1_row = m1[np.newaxis,:]
m2_col = m2[:,np.newaxis]
m2_row = m2[np.newaxis,:]
print(m1,m2,m1_col.shape,m1_row.shape)



X1 = np.matrix(X1).T
X2 = np.matrix(X2).T
c1 = np.matrix(np.cov(X1))
c2 = np.matrix(np.cov(X2))
c1_inv = np.linalg.inv(c1)
c2_inv = np.linalg.inv(c2)
c1_det = np.linalg.det(c1)
c2_det = np.linalg.det(c2)
W1 = -1/2*c1_inv
W2 = -1/2*c2_inv
w1 = c1_inv*m1_col
w2 = c2_inv*m2_col
w10 = -1/2*np.dot(np.dot(m1_row,c1_inv),m1_col) - 1/2*log(c1_det) + log(priors[0])
w20 = -1/2*np.dot(np.dot(m2_row,c2_inv),m2_col) - 1/2*log(c2_det) + log(priors[1])

# print(W1.shape,w1.shape,w10.shape)
W12 = W1 - W2
w12 = w1 - w2

print(W1,w1,w10,W2,w2,w20)


print("%f x1^2 + %f x2^2 + %f x1*x2 + %f x1 + %f x2 + %f = 0" % 
    (W12[0, 0], W12[1, 1], W12[0, 1] + W12[1, 0],w12[0, 0], w12[1, 0], w10-w20))




x = np.arange(0,6,0.01)
y = np.arange(-10,10,0.01)
x,y = np.meshgrid(x,y)
# -0.562500 x1^2 + 0.000000 x2^2 + 0.000000 x1*x2 + 3.375000 x1 + 3.000000 x2 + -10.369353 = 0
z = -0.562500*np.power(x,2) + 3.375000*x + 3*y - 10.369353
plt.contour(x,y,z,0)
plt.savefig('nb.png')