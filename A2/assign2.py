import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import det_curve

# def fun1(x1,x2,u,E):
#     u = u.reshape((2,1))
#     x = np.array([x1,x2]).reshape((2,1))
#     return 1/2/np.pi/np.linalg.det(E) * np.exp(-1/2*np.dot((x-u).T,np.dot(np.linalg.inv(E),(x-u))))

# def discrimant1(E,u,P,x,y):
#     x = np.array([x,y])
#     E_inv = np.linalg.inv(E)
#     W = -1/2 * E_inv
#     w = np.dot(E_inv,u)
#     w0 = -1/2*np.dot(np.dot(u.T,E_inv),u) - 1/2 * np.log(np.linalg.det(E)) + np.log(P)
#     g_x = np.dot(np.dot(x.T,W),x) + np.dot(w.T,x) + w0
#     return g_x
# Z1=np.zeros((len(X),len(X[0])))
    # Z2=np.zeros((len(X),len(X[0])))
    # Z3=np.zeros((len(X),len(X[0])))
    # for i in range(len(X)):
    #     for j in range(len(X[0])):
    #         Z1[i][j]=discrimant(cov1,mean1,P1,X[i][j],Y[i][j])
    #         Z2[i][j]=discrimant(cov2,mean2,P2,X[i][j],Y[i][j])
    #         Z3[i][j]=discrimant(cov3,mean3,P3,X[i][j],Y[i][j])


    # classify=np.zeros((len(X),len(X[0])))
    # for i in range(len(X)):
    #     for j in range(len(X[0])):
    #         classify[i][j] = np.argmax([Z1[i][j],Z2[i][j],Z3[i][j]])+1
def PDF(X,Y,u,E):
    _E = np.linalg.inv(E)
    _X = X - u[0]
    _Y = Y - u[1] 
    exp_arg = -1/2 * (_E[0][0] * _X * _X + (_E[0][1] + _E[1][0]) * _X * _Y + _E[1][1] * _Y * _Y)
    return (1/(2*np.pi*np.sqrt(np.linalg.det(E)))) * np.exp(exp_arg)   

def discrimant(E,u,P,_X,_Y):
    E_inv = np.linalg.inv(E)
    W = 1/2 * E_inv
    w = np.dot(E_inv,u)
    w0 = -1/2*np.dot(np.dot(u.T,E_inv),u) - 1/2 * np.log(np.linalg.det(E)) + np.log(P)
    g_x = -(W[0][0] * _X * _X + (W[0][1] + W[1][0]) * _X * _Y + W[1][1] * _Y * _Y) + (w[0] * _X + w[1] * _Y) + w0
    return g_x

def mean(x):
    return sum(x)/len(x)

def covariance(x,mean):
    out = np.zeros((len(x[0]),len(x[0])))
    for v in x:
        vec = (v-mean)
        vec = vec.reshape((len(vec),1))
        out += np.dot((vec),vec.T)
    return 1/(len(x)-1) * out

def sigma(x,mean):
    out = np.sum((x-mean)*(x-mean))
    return 1/(len(x)-1) * out

def classification(means,covs,tst_data,P):
    X,Y,T = tst_data[:,0],tst_data[:,1],tst_data[:,2]
    Z1 = discrimant(covs[0],means[0],P[0],X,Y)
    Z2 = discrimant(covs[1],means[1],P[1],X,Y)
    Z3 = discrimant(covs[2],means[2],P[2],X,Y)
    classify = np.array([Z1,Z2,Z3]).argmax(0)
    return((len(T)-np.count_nonzero(classify-T+1))/len(T) * 100)

def plots(means,covs,class1,class2,class3,data,tst_data,P,flag=0):
    mins = [min(data[:,0]),min(data[:,1])]
    maxs = [max(data[:,0]),max(data[:,1])]
    
    tst_class1,tst_class2,tst_class3=[],[],[]
    for i in range(len(tst_data)):
        if tst_data[i][2]==1:
            tst_class1.append(tst_data[i][:2])
        if tst_data[i][2]==2:
            tst_class2.append(tst_data[i][:2])
        if tst_data[i][2]==3:
            tst_class3.append(tst_data[i][:2])
    tst_class1=np.array(tst_class1)
    tst_class2=np.array(tst_class2)
    tst_class3=np.array(tst_class3)

    n = 500
    x1 = np.linspace(mins[0]-3,maxs[0]+3,n)
    x2 = np.linspace(mins[1]-3,maxs[1]+3,n)
    X,Y = np.meshgrid(x1,x2)

    Z1 = PDF(X,Y,means[0],covs[0])
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z1,cmap='viridis')

    Z2 = PDF(X,Y,means[1],covs[1])    
    ax.plot_surface(X, Y, Z2,cmap='viridis')

    Z3 = PDF(X,Y,means[2],covs[2])
    ax.plot_surface(X, Y, Z3,cmap='viridis')
    plt.show()

    fig, ax = plt.subplots(1,1)
    ax.contour(X,Y,Z1)
    ax.contour(X,Y,Z2)
    ax.contour(X,Y,Z3)

    Z1 = discrimant(covs[0],means[0],P[0],X,Y)
    Z2 = discrimant(covs[1],means[1],P[1],X,Y)
    Z3 = discrimant(covs[2],means[2],P[2],X,Y)
    classify=np.zeros((len(X),len(X[0])))
    for i in range(len(X)):
        for j in range(len(X[0])):
            classify[i][j] = np.argmax([Z1[i][j],Z2[i][j],Z3[i][j]])+1
    ax.contourf(X, Y, classify)
  
    ax.set_title('Contour Plot')
    ax.set_xlabel('feature_x')
    ax.set_ylabel('feature_y')

    if flag == 0:
        ax.scatter(class1[:,0],class1[:,1],facecolors='none',edgecolors='r')
        ax.scatter(class2[:,0],class2[:,1],facecolors='none',edgecolors='b')
        ax.scatter(class3[:,0],class3[:,1],facecolors='none',edgecolors='g')
    elif flag == 1:

        ax.scatter(tst_class1[:,0],tst_class1[:,1],facecolors='none',edgecolors='r')
        ax.scatter(tst_class2[:,0],tst_class2[:,1],facecolors='none',edgecolors='b')
        ax.scatter(tst_class3[:,0],tst_class3[:,1],facecolors='none',edgecolors='g')
    plt.show()

    #ROC 
    S = []
    S.append(PDF(tst_data[:,0],tst_data[:,1],means[0],covs[0]) * P[0])
    S.append(PDF(tst_data[:,0],tst_data[:,1],means[1],covs[1]) * P[1])
    S.append(PDF(tst_data[:,0],tst_data[:,1],means[2],covs[2]) * P[2])
    S = np.array(S).T
    Scores = sorted(S.reshape((len(tst_data)*3,1)).flatten())

    TPR = [0]*len(Scores)
    FPR = [0]*len(Scores)
    count=0
    for threshold in Scores:
        TP,FP,TN,FN = 0,0,0,0
        for i in range(len(tst_data)):
            for j in range(3):
                if S[i][j] >= threshold:
                    if tst_data[i][2] == j+1: TP+=1
                    else:   FP+=1
                else:
                    if tst_data[i][2] == j+1: FN+=1
                    else:   TN+=1
        TPR[count] = TP/(TP+FN)
        FPR[count] = FP/(FP+TN)
        count+=1
    plt.figure()
    plt.plot(FPR,TPR)
    plt.show()

    #DET
    S = (S.reshape((len(tst_data)*3,1))).flatten()
    y_true = []
    for i in range(3):
        for j in range(len(tst_data)):
            if tst_data[j][2] == i+1:
                y_true.append(1)
            else:
                y_true.append(0)

    y_true = np.array(y_true)
    fpr, fnr, thresholds = det_curve(y_true, S)
    plt.figure()
    plt.plot(fpr,fnr)
    plt.show()

def case_1_4(data,tst_data,class1,class2,class3,means,P):

    cov1 = covariance(class1[:,:2],means[0])
    cov2 = covariance(class2[:,:2],means[1])
    cov3 = covariance(class3[:,:2],means[2])
    cov = 1/(len(data)-3) * (cov1 * (len(class1)+1) + cov2 * (len(class2)+1) + cov3 * (len(class3)+1))
    covs = [cov,cov,cov]
    plots(means,covs,class1,class2,class3,data,tst_data,P)

    accuracy = classification(means,covs,tst_data,P)
    print("Case 1 : " ,accuracy)

    #CASE 4
    cov = np.diag(np.diag(cov))
    covs = [cov,cov,cov]
    plots(means,covs,class1,class2,class3,data,tst_data,P)

    accuracy = classification(means,covs,tst_data,P)
    print("Case 4 : " ,accuracy)

def case_3(data,tst_data,class1,class2,class3,means,P):
    
    cov1 = sigma(class1[:,:2],means[0]) * np.diag([1,1])
    cov2 = sigma(class2[:,:2],means[1]) * np.diag([1,1])
    cov3 = sigma(class3[:,:2],means[2]) * np.diag([1,1])
    covs = [cov1,cov2,cov3]

    plots(means,covs,class1,class2,class3,data,tst_data,P)

    accuracy = classification(means,covs,tst_data,P)
    print("Case 3 : " ,accuracy)


def case_2_5(data,tst_data,class1,class2,class3,means,P):
    #CASE 2

    cov1 = covariance(class1[:,:2],means[0])
    cov2 = covariance(class2[:,:2],means[1])
    cov3 = covariance(class3[:,:2],means[2]) 
    covs = [cov1,cov2,cov3]

    plots(means,covs,class1,class2,class3,data,tst_data,P)

    accuracy = classification(means,covs,tst_data,P)
    print("Case 2 : " ,accuracy)

    #CASE 5
    cov1 = np.diag(np.diag(cov1))
    cov2 = np.diag(np.diag(cov2))
    cov3 = np.diag(np.diag(cov3))
    covs = [cov1,cov2,cov3]

    plots(means,covs,class1,class2,class3,data,tst_data,P)

    accuracy = classification(means,covs,tst_data,P)
    print("Case 5 : " ,accuracy)

def func(data,tst_data):
    class1,class2,class3=[],[],[]
    for i in range(len(data)):
        if data[i][2]==1:
            class1.append(data[i][:2])
        if data[i][2]==2:
            class2.append(data[i][:2])
        if data[i][2]==3:
            class3.append(data[i][:2])
    class1=np.array(class1)
    class2=np.array(class2)
    class3=np.array(class3)

    mean1 = mean(class1[:,:2])
    mean2 = mean(class2[:,:2])
    mean3 = mean(class3[:,:2])
    means = [mean1,mean2,mean3]

    
    P1 = len(class1)/len(data)
    P2 = len(class2)/len(data)
    P3 = len(class3)/len(data)
    P = [P1,P2,P3]

    
    case_3(data,tst_data,class1,class2,class3,means,P)
    case_1_4(data,tst_data,class1,class2,class3,means,P)
    case_2_5(data,tst_data,class1,class2,class3,means,P)
    
"""with open("data/Linsep/trian.txt") as f:
    train_data = [[float(val) for val in line.strip().split(',')] for line in f]
train_data = np.array(train_data)
with open("data/Linsep/dev.txt") as f:
    tst_data = [[float(val) for val in line.strip().split(',')] for line in f]
tst_data = np.array(tst_data)
func(train_data,tst_data)"""

with open("data/Nonsep/trian.txt") as f:
    train_data = [[float(val) for val in line.strip().split(',')] for line in f]
train_data = np.array(train_data)
with open("data/Nonsep/dev.txt") as f:
    tst_data = [[float(val) for val in line.strip().split(',')] for line in f]
tst_data = np.array(tst_data)
func(train_data,tst_data)

"""with open("data/RealData/trian.txt") as f:
    train_data = [[float(val) for val in line.strip().split(',')] for line in f]
train_data = np.array(train_data)
with open("data/RealData/dev.txt") as f:
    tst_data = [[float(val) for val in line.strip().split(',')] for line in f]
tst_data = np.array(tst_data)
func(train_data,tst_data)"""