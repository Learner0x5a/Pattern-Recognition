'''
梯度下降法
'''
import numpy as np 
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
fig = plt.figure()

W1 = np.array([(1,1),(2,0),(2,1),(0,2),(1,3)])
W2 = np.array([(-1,2),(0,0),(-1,0),(-1,-1),(0,-2)])
Z1 = np.concatenate((W1,np.ones(W1.shape[0])[:,np.newaxis]),axis=1)
Z2 = np.concatenate((W2,np.ones(W2.shape[0])[:,np.newaxis]),axis=1)
X = np.concatenate((Z1,-Z2),axis=0)

# Y = np.concatenate((np.zeros(5),np.ones(5)),axis=0)
print(X)
plt.scatter(W1[:,0],W1[:,1],label='W1')
plt.scatter(W2[:,0],W2[:,1],label='W2')
plt.xlabel('x')
plt.ylabel('y')


def linear_neuron(w,x):
    y = np.sum(np.multiply(w,x))
    return y

w = np.random.randn(3)
lr = 1e-2
epochs = 1000
for i in range(epochs):
    for x in X:    
        if linear_neuron(w,x) > 0:
            continue
        else:
            w = w + lr*x
print('{:.2f}*x + {:.2f}*y + {:.2f} = 0'.format(w[0],w[1],w[2]))
X =  X = np.concatenate((W1,W2),axis=0)
line_x = np.linspace(-1,2)
'''
w0*x + w1*y + w2 = 0  <->  y = -1/w1 * (w0*x + w2)
'''
line_y = -1/w[1] * (w[0]*line_x + w[2])
plt.plot(line_x,line_y,c='r',label='{:+.2f}*x {:+.2f}*y {:+.2f} = 0'.format(w[0],w[1],w[2]))
plt.legend()
plt.savefig('gd.png')