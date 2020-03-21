import numpy as np 
import matplotlib
matplotlib.use('AGG')#或者PDF, SVG或PS
import matplotlib.pyplot as plt

W1 = np.array([(2,3), (2,2), (2,4), (3,3), (3,4), (2.5,3), (1.5,2), (3.5,2.5), (4,4), (0.5,0.5)])
W2 = np.array([(0,2.5), (-2,2), (-1,-1), (1,-2), (3,0), (-2,-2), (-3,-4), (-5,-2), (4,-1)])
M1 = np.mean(W1,axis=0)
M2 = np.mean(W2,axis=0)
x1 = M1[0]
x2 = M2[0]
y1 = M1[1]
y2 = M2[1]

line_x = np.linspace(-4,4,20)
line_y = (y1+y2)/2.+-1/((y2-y1)/(x2-x1))*(line_x-(x1+x2)/2)
print('y = (y1+y2)/2+-1/((y2-y1)/(x2-x1))*(x-(x1+x2)/2) = '+str((y1+y2)/2.)+'+'+str(-1/((y2-y1)/(x2-x1)))+'*(x-'+str((x1+x2)/2)+')')



W = np.vstack((W1,W2))
print(M1,M2)
x1 = W1[:,0]
y1 = W1[:,1]
x2 = W2[:,0]
y2 = W2[:,1]
# print(W,x,y)



plt.xlabel(u'x',FontSize=16)
plt.ylabel(u'y',FontSize=16)
# plt.title(img,fontsize='large',fontweight='bold')
plt.scatter(x1,y1,c='black',marker='x',label='W1')
plt.scatter([M1[0]],[M1[1]],c='blue',marker='x',label='M1')
plt.scatter(x2,y2,c='red',marker='s',label='W2')
plt.scatter([M2[0]],[M2[1]],c='green',marker='s',label='M2')
plt.plot(line_x,line_y,color='orange',linestyle='-.',label='separating plane')

plt.legend()
plt.savefig("dist.png")
# plt.clf()
