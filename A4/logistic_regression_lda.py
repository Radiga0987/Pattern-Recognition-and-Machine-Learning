import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from scipy import signal
import math

onehot_encoder = OneHotEncoder(sparse=False)
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from statistics import mode
from pca_lda_ import pca_,lda_,transform

def gradient(X, T, W, mu):
        Z = np.dot(X,W)
        Y = softmax(Z, axis=1)
        N = X.shape[0]
        gd = 1/N * np.dot(X.T,(Y - T)) + 2 * mu * W
        return gd

def logistic_reg(X, T, max_iter=10000, eta=0.8, mu=0):
    T = onehot_encoder.fit_transform(T.reshape(-1,1))
    W = np.zeros((X.shape[1], T.shape[1]))
    step = 0
    while step < max_iter:
        step += 1
        W -= eta * gradient(X, T, W, mu)
    return W

def predict(H,W):
    Z = np.dot(H,W)
    P = softmax(Z, axis=1)
    return np.argmax(P, axis=1)


# def covariance(x,u):
#     N = len(x)-1
#     out = np.zeros((len(x[0]),len(x[0])))
#     for i,v in enumerate(x):
#         vec = (v-u)
#         vec = vec.reshape((len(vec),1))
#         out += np.dot(vec,vec.T)
#     covariance_matrix = (1/N) * out
#     return covariance_matrix

# def lda_(data,class_labels,no_classes,L):
#     no_features = len(data[0])
#     Sw = np.zeros((no_features,no_features))
#     Sb = np.zeros((no_features,no_features))
#     mean = np.mean(data,axis=0)
#     for cl in range(no_classes):
#         indexes = np.where(class_labels == cl+1) 
#         dat = data[indexes]
#         Sw += len(dat)*covariance(dat,np.mean(dat,axis=0))
#         temp = np.mean(dat,axis=0) - mean
#         temp = temp.reshape((len(temp),1))
#         Sb += len(dat) * np.dot(temp,temp.T)
    
#     e,V = np.linalg.eig(np.dot(np.linalg.inv(Sw),Sb))
#     e,V = map(np.array, zip(*sorted(zip(e,V.T),reverse=True)))      #Sorting
#     V=V.T
#     for i in range(len(e)):
#         exp_var = np.sum(e[:i+1]) / np.sum(e)
#         if exp_var > L:
#             break
#     return V[:,:i+1]
    
# def lda_transform(data,Q):
#     data_transform = np.dot(Q.T,data.T)
#     return data_transform.T    

# def fit(X, y, Per):
#     n_features = X.shape[1]
#     class_labels = np.unique(y)

#     # Within class scatter matrix:
#     # SW = sum((X_c - mean_X_c)^2 )

#     # Between class scatter:
#     # SB = sum( n_c * (mean_X_c - mean_overall)^2 )

#     mean_overall = np.mean(X, axis=0)
#     SW = np.zeros((n_features, n_features))
#     SB = np.zeros((n_features, n_features))
#     for c in class_labels:
#         X_c = X[y == c]
#         mean_c = np.mean(X_c, axis=0)
#         # (4, n_c) * (n_c, 4) = (4,4) -> transpose
#         SW += (X_c - mean_c).T.dot((X_c - mean_c))

#         # (4, 1) * (1, 4) = (4,4) -> reshape
#         n_c = X_c.shape[0]
#         mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
#         SB += n_c * (mean_diff).dot(mean_diff.T)

#     # Determine SW^-1 * SB
#     A = np.linalg.inv(SW).dot(SB)
#     # Get eigenvalues and eigenvectors of SW^-1 * SB
#     eigenvalues, eigenvectors = np.linalg.eig(A)
#     # -> eigenvector v = [:,i] column vector, transpose for easier calculations
#     # sort eigenvalues high to low
#     eigenvectors = eigenvectors.T
#     idxs = np.argsort(abs(eigenvalues))[::-1]
#     eigenvalues = eigenvalues[idxs]
#     eigenvectors = eigenvectors[idxs]
#     eigenvectors = eigenvectors.T
#     # store first n eigenvectors
#     for i in range(len(eigenvalues)):
#         exp_var = np.sum(eigenvalues[:i+1]) / np.sum(eigenvalues)
#         if exp_var > Per:
#             break
#     return eigenvectors[:, : Per+1]

# def transform(Q, X):
#     # project data
#     return np.dot(X, Q).real

# def pca_(data,L):
#     cov = covariance(data,np.mean(data,axis=0))
#     e,V = np.linalg.eig(cov)        #Finding Eigen values and vectors of martix A
#     mag,e,V = map(np.array, zip(*sorted(zip(abs(e),e,V.T),reverse=True)))   #Sorting eigen vals and vecs
#     V=V.T
#     for i in range(len(cov)):
#         exp_var = np.sum(e[:i+1]) / np.sum(e)
#         if exp_var > L:
#             break
#     return V[:,:i+1]
#     return V[:,:L]

# def pca_transform(data,Q):
#     data_transform = np.dot(Q.T,data.T)
#     return data_transform.T

def confusion_matrix(conf_matrix,Title=""):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.6)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='large')
    plt.xlabel('Predictions', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title(Title, fontsize=12)
    plt.show()


def Synth():
    no_classes = 2

    with open("Synthetic_Dataset/train.txt") as f:
            train_data = [[float(val) for val in line.strip().split(',')] for line in f]
    train_data = np.array(train_data)
    with open("Synthetic_Dataset/dev.txt") as f:
        dev_data = [[float(val) for val in line.strip().split(',')] for line in f]
    dev_data = np.array(dev_data)

    #Separating Train data into respective classes
    class1,class2=[],[]
    for i in range(len(train_data)):
        if train_data[i][2]==1:
            class1.append(train_data[i][:2])
        if train_data[i][2]==2:
            class2.append(train_data[i][:2])
    class1=np.array(class1)
    class2=np.array(class2)

    sc_x = StandardScaler()    

    #Order of Polynomial
    n = 30
    x1,x2=train_data[:,0],train_data[:,1]
    phiT=[]
    #Phi matrix is constructed
    for count in range(n+1):
        for i in range(count+1):
            phiT.append((x1**(count-i))*(x2**i))
    phi = np.array(phiT).T

    #Mean Normalize
    phi = sc_x.fit_transform(phi)

    #Find Weights
    W = logistic_reg(phi,train_data[:,2],10000,0.8,0)

    x1=dev_data[:,0]
    x2=dev_data[:,1]
    phiT=[]
    #Phi matrix is constructed
    for count in range(n+1):
        for i in range(count+1):
            phiT.append((x1**(count-i))*(x2**i))
    phi = np.array(phiT).T

    #Mean Normalize
    phi = sc_x.transform(phi)

    #Predict
    y_pred = 1+predict(phi,W)
    print("Accuracy of Classification",(1-sum(abs(y_pred-dev_data[:,2]))/len(dev_data))*100,"%")
    
    conf_matrix = np.zeros((no_classes,no_classes))
    for i in range(len(dev_data)):
        conf_matrix[int(dev_data[:,2][i]-1)][int(y_pred[i]-1)] += 1
    confusion_matrix(conf_matrix,"Synthetic Dataset Logistic Regression")
        

    #Decision Boundary Plot
    x1_ = np.linspace(-17, 17,500)
    x2_ = np.linspace(-17, 17,500)
    X,Y = np.meshgrid(x1_,x2_)
    Space = [[] for i in range(len(X)*len(X))]
    count = 0
    for i in range(len(X)):
        for j in range(len(X[0])):
            Space[count] = [X[i][j],Y[i][j]]
            count+=1
    Space = np.array(Space)

    x1=Space[:,0]
    x2=Space[:,1]
    phiT=[]
    #Phi matrix is constructed
    for count in range(n+1):
        for i in range(count+1):
            phiT.append((x1**(count-i))*(x2**i))
    phi = np.array(phiT).T
    phi = sc_x.transform(phi)
    y_pred = 1+predict(phi,W)
    val_ = y_pred.reshape(500,500)

    plt.figure(figsize=(7,7))
    ax = plt.subplot()
    ax.plot(class1[:,0],class1[:,1],'.',label="Class1  - yellow")
    ax.plot(class2[:,0],class2[:,1],'.',label="Class2 - red")
    c = ax.contourf(X, Y, val_,cmap='YlOrRd',levels=[0,1,2])
    plt.show()

def Image_Dataset():
    no_classes = 5
    classes = ['coast','forest','highway','mountain','opencountry']
    train = []  
    train_all = []
    train_labels = []
    dev_labels= []
    dev = []
    test = []
    denoms = []
    priors = np.zeros(len(classes))

    #Reading the Images of each class for train and Developement DataSets
    for i,cls in enumerate(classes):
        lst = []
        dir_list = os.listdir('Features//'+cls+'//train')
        priors[i] = len(dir_list)
        for file in dir_list:
            train_labels.append(i+1)            #train_labels.extend([i+1]*36)
            image =  np.loadtxt('Features/'+cls+'/train/' + file)          #lst.extend(np.loadtxt('Features/'+cls+'/train/' + file))
            train_all.append(image.flatten())
        # lst = np.array(lst)
        # train_all.extend(lst)

        lst = []
        lst2 = []
        dir_list = os.listdir('Features//'+cls+'//dev')
        for file in dir_list:
            dev_labels.append(i+1)
            image =  np.loadtxt('Features/'+cls+'/dev/' + file)
            dev.append(image.flatten())
    
    train_all = np.array(train_all)
    print(train_all.shape)
    train_labels = np.array(train_labels)

    sc_x = StandardScaler()    
    train_all = sc_x.fit_transform(train_all)
    dim_old = len(train_all[0])
    # Q = lda_(train_all,train_labels,99)
    Q = pca_(train_all,0.99)
    train_all = transform(Q,train_all)
    dim_new = len(train_all[0])
    print("Pca Transformed ",dim_old," to ",dim_new," features")

    # plt.scatter(
    #     train_all[:,0], train_all[:,1], c=train_labels, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 5)
    # )
    # plt.show()

    sc_x2 = StandardScaler()
    train_all = sc_x2.fit_transform(train_all)
    train_all = np.hstack((np.ones((len(train_all),1)),train_all))

    W = logistic_reg(train_all,train_labels,10000,0.8,1e-3)

    count_correct = 0
    count_dev_files = 0
    S = []
    class_labels = []

    data = sc_x.transform(dev)
    data = transform(Q,data)
    # plt.scatter(
    #     data[:,0], data[:,1], c=dev_labels, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 5)
    # )
    # plt.show()
    # data = lda_transform(data,Q)
    data = sc_x2.transform(data)
    data = np.hstack((np.ones((len(data),1)),data))
    y_pred = 1+predict(data,W)
    y_pred = np.array(y_pred)
    print(y_pred)
    print("Accuracy of Classification",(1-np.count_nonzero(y_pred-dev_labels)/len(dev_labels))*100,"%")

    conf_matrix = np.zeros((no_classes,no_classes))
    for i in range(len(dev_labels)):
        conf_matrix[int(dev_labels[i]-1)][int(y_pred[i]-1)] += 1
    confusion_matrix(conf_matrix,"Image Dataset Logistic Regression")

def spoken_digits():
    # Isolated Digits
    digits = [1,2,5,9,'z']
    no_classes = 5
    train = []
    dev = []

    #Loading train and dev data
    for digit in digits:
        lst = []
        dir_list = os.listdir('Isolated_Digits//'+str(digit)+'//train')
        for file in dir_list:
            if file.endswith('.mfcc'):
                lst.append(np.loadtxt('Isolated_Digits/'+str(digit)+'/train/' + file,skiprows = 1))

        train.append(np.array(lst,dtype=object))

        lst = []
        dir_list = os.listdir('Isolated_Digits//'+str(digit)+'//dev')
        for file in dir_list:
            if file.endswith('.mfcc'):
                lst.append(np.loadtxt('Isolated_Digits/'+str(digit)+'/dev/' + file,skiprows = 1))

        dev.append(np.array(lst,dtype=object))

    count = 0
    num_frames = 0
    for cls in range(len(train)):
        for i in range(len(train[cls])):
            num_frames += len(train[cls][i])
            count += 1
    avg_num_frames = math.floor(num_frames/count)

    train_all = []
    dev_all = []
    train_labels = []
    dev_labels = []
    for cls in range(len(train)):
        for i in range(len(train[cls])):
            train_all.append(signal.resample(train[cls][i],avg_num_frames))
            train_labels.append(cls+1)

    for cls in range(len(dev)):
        for i in range(len(dev[cls])):
            dev_all.append(signal.resample(dev[cls][i],avg_num_frames))
            dev_labels.append(cls+1)

    train_all = np.array(train_all)
    train_labels = np.array(train_labels)
    dev_all = np.array(dev_all)

    mean_train = np.mean(train_all,axis=0)
    maxs = np.max(train_all,axis=0)
    mins = np.min(train_all,axis=0)
    denoms = maxs - mins
    train_all = (train_all-mean_train)/denoms
    dev_all = (dev_all-mean_train)/denoms

    train_extended = []
    dev_extended = []
    for i in range(len(train_all)):
        lst = []
        for j in range(len(train_all[i])):
            lst.extend(train_all[i][j])
        train_extended.append(np.array(lst))
    train_extended =  np.array(train_extended)
    for i in range(len(dev_all)):
        lst = []
        for j in range(len(dev_all[i])):
            lst.extend(dev_all[i][j])
        dev_extended.append(np.array(lst))
    dev_extended = np.array(dev_extended)

    dim_old = len(train_extended[0])
    Q = lda_(train_extended,0.99)
    train_extended = transform(Q,train_extended)
    dim_new = len(train_extended[0])
    print("Lda Transformed ",dim_old," to ",dim_new," features")

    
    W = logistic_reg(train_extended,train_labels,10000,0.8,0)

    dev_extended = transform(Q,dev_extended)
    y_pred = 1+predict(dev_extended,W)
    y_pred = np.array(y_pred)
    
    print("Accuracy of Classification",(1-np.count_nonzero(y_pred-dev_labels)/len(dev_labels))*100,"%")

    conf_matrix = np.zeros((no_classes,no_classes))
    for i in range(len(dev_labels)):
        conf_matrix[int(dev_labels[i]-1)][int(y_pred[i]-1)] += 1
    confusion_matrix(conf_matrix,"Image Dataset Logistic Regression")
# Synth()
Image_Dataset()
# spoken_digits()