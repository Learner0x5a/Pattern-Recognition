import numpy as np 
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
fig = plt.figure()

W1 = np.array([(2,0),(2,2),(2,4),(3,3)])
W2 = np.array([(0,3),(-2,2),(-1,-1),(1,-2),(3,-1)])
W3 = np.array([(1,1),(2,0),(2,1),(0,2),(1,3)])
W4 = np.array([(-1,2),(0,0),(-1,0),(-1,1),(0,-2)])

def main(W1,W2):
    X = np.concatenate((W1,W2),axis=0)
    Y = np.concatenate((np.zeros(W1.shape[0]),np.ones(W2.shape[0])),axis=0)
    print(X,Y)
    plt.scatter(W1[:,0],W1[:,1],label='W1')
    plt.scatter(W2[:,0],W2[:,1],label='W2')
    plt.xlabel('x')
    plt.ylabel('y')

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import mean_squared_error
    model = LinearDiscriminantAnalysis()
    model.fit(X,Y)
    mse = mean_squared_error(Y,model.predict(X))
    print('mse:',mse)
    print(model.intercept_,model.coef_)
    w = model.coef_[0]
    line_x = np.linspace(-2,3)
    line_y = -1/w[1] * (w[0]*line_x + model.intercept_)
    plt.plot(line_x,line_y,c='r',label='{:+.2f}*x {:+.2f}*y {:+.2f} = 0; mse={:.2f}'.format(w[0],w[1],model.intercept_[0],mse))
    plt.legend()
    plt.savefig('lda.png')

main(W1,W2)
# main(W3,W4)