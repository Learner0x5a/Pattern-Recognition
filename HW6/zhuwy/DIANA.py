from math import log # natual log
import numpy as np 
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
fig = plt.figure()

priors = [0.5,0.5]
X1 = np.array([(2,0),(2,2),(2,4),(3,3)])
X2 = np.array([(0,3),(-2,2),(-1,-1),(1,-2),(3,-1)])
X = np.concatenate((X1,X2),axis=0)
Y = np.concatenate((np.zeros(X1.shape[0]),np.ones(X2.shape[0])),axis=0)
plt.scatter(X1[:,0],X1[:,1],label='W1')
plt.scatter(X2[:,0],X2[:,1],label='W2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
# print(X1,X2)

m1 = np.mean(X1,axis=0)
m2 = np.mean(X2,axis=0)
m1_col = m1[:,np.newaxis]
m1_row = m1[np.newaxis,:]
m2_col = m2[:,np.newaxis]
m2_row = m2[np.newaxis,:]
# print(m1,m2,m1_col.shape,m1_row.shape)



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

# -2.009301 x1^2 + -0.047216 x2^2 + 0.530450 x1*x2 + 8.897630 x1 + -0.087204 x2 + -8.613675 = 0
x = np.arange(-3,4,0.01)
y = np.arange(-3,5,0.01)
x,y = np.meshgrid(x,y)

z = -2.009301*np.power(x,2) - 0.047216*np.power(y,2) + 0.530450*np.multiply(x,y) + 8.89763*x - 0.087204*y - 8.613675
plt.contour(x,y,z,0)
# plt.savefig('nb.png')


from sklearn.metrics import euclidean_distances
def dist_matrix(X):
    m,n = np.shape(X)
    matrix = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            matrix[i][j] = euclidean_distances([X[i]],[X[j]])[0]
    return matrix

distance_matrix = dist_matrix(X)
# print(distance_matrix.shape)
points = []

for x,y in X:
    points.append('('+str(x)+','+str(y)+')')
# print(points)
import pandas as pd
distance_matrix = pd.DataFrame(distance_matrix,index=points, columns=points) # trick：用label做索引

def avg_dist_within_group_element(point, group):
    max_diameter = -np.inf
    sum_dist = 0
    for i in group:
        sum_dist += distance_matrix[point][i] 
        if(distance_matrix[point][i] > max_diameter):
            max_diameter = distance_matrix[point][i]
    if len(group) > 1:
        avg_dist = sum_dist/(len(group)-1)
    else: 
        avg_dist = 0
    return avg_dist

def avg_dist_across_group_element(point, main_group, splinter_group):
    if len(splinter_group) == 0:
        return 0
    sum_dist = 0
    for j in splinter_group:
        sum_dist = sum_dist + distance_matrix[point][j]
    avg_dist = sum_dist/(len(splinter_group))
    return avg_dist
    
    
def splinter(main_group, splinter_group):
    max_distance = -np.inf
    max_distance_point = None
    for point in main_group:
        x = avg_dist_within_group_element(point, main_group)
        y = avg_dist_across_group_element(point, main_group, splinter_group)
        diff = x - y
        if diff > max_distance:
            max_distance = diff
            max_distance_point = point
    if max_distance > 0:
        return (max_distance_point, True)
    else:
        return (-1, False)
    
def split(group):
    main_group = group
    splinter_group = []
    (max_distance_point,is_splinter) = splinter(main_group, splinter_group)
    while is_splinter:
        main_group.remove(max_distance_point)
        splinter_group.append(max_distance_point)
        (max_distance_point,is_splinter) = splinter(group, splinter_group)
    
    return (main_group, splinter_group)

def max_diameter(cluster_list): # 求各个类中直径最大的
    max_diameter_cluster_point = None
    max_diameter_cluster_value = -np.inf
    index = 0
    for group in cluster_list:
        for i in group:
            for j in group:
                if distance_matrix[i][j] > max_diameter_cluster_value:
                    max_diameter_cluster_value = distance_matrix[i][j]
                    max_diameter_cluster_point = index
        index +=1

    if(max_diameter_cluster_value <= 0):
        return -1
    return max_diameter_cluster_point

def diameter(clu): # 求类的直径的两个端点
    value_clu = []
    for pt_str in clu:
        pt_str = pt_str[1:-1].split(',')
        value_clu.append((float(pt_str[0]),float(pt_str[1])))
    # print(value_clu)
    dia = -np.inf # 直径
    endpt1 = None # 端点1
    endpt2 = None # 端点2
    for pt1 in value_clu:
        for pt2 in value_clu:
            dist = euclidean_distances([pt1],[pt2])[0]
            if dist > dia:
                dia = dist
                endpt1 = pt1
                endpt2 = pt2     
    return np.array(endpt1),np.array(endpt2),dia


# fig = plt.figure()

R = ([points]) # current clusters
hierarchy = 1
index = 0
print(hierarchy, R)
clr = ['r','g','b','c','m','y','k','w','navy','peru','azure']
while(index!=-1):
    (A, B) = split(R[index])
    del R[index]
    R.append(A)
    R.append(B)
    index = max_diameter(R)
    hierarchy += 1
    print(hierarchy, R)
    # 求各个类的端点画圆。
    for clu in R:
        if len(clu) > 1: # 只有一个点的跳过
            endpt1,endpt2,dia = diameter(clu)
            circle1 = plt.Circle(xy = (endpt1 + endpt2)/2.,color=clr[hierarchy], radius=dia/2+0.1, alpha=0.1)
            plt.gcf().gca().add_artist(circle1)


plt.savefig('hierarchy.png')




