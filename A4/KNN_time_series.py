import numpy as np
import os
import time
from sklearn.decomposition import PCA
from scipy import signal
import math


def get_dist_to_points(train,dev):
    dev_square = np.sum(np.square(dev),axis=1,keepdims=1)
    b = np.ones((1,train.shape[0]))
    dists = dev_square.dot(b)
    dev_square = np.ones((dev.shape[0],1))
    b = np.sum(np.square(train),axis=1,keepdims=1).T
    dists += dev_square.dot(b)
    dists -= 2*dev.dot(train.T)
    dists = np.sqrt(dists)
    return dists

def predict_KNN(train,train_labels,dev,k):
    dists_matrix = get_dist_to_points(train,dev)
    indices = np.argpartition(dists_matrix, k-1, axis=1)[:, :k]

    predictions = []
    for i in range(len(indices)):
            count_k = [0]*len(set(train_labels))
            for j in range(k):
                count_k[train_labels[indices[i][j]]] += 1
            
            predictions.append(count_k.index(max(count_k)))
    return predictions

# Isolated Digits
digits = [1,2,5,9,'z']
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
avg_num_frames = 200
train_all = []
dev_all = []
train_labels = []
dev_labels = []
for cls in range(len(train)):
    for i in range(len(train[cls])):
        train_all.append(signal.resample(train[cls][i],avg_num_frames))
        train_labels.append(cls)

for cls in range(len(dev)):
    for i in range(len(dev[cls])):
        dev_all.append(signal.resample(dev[cls][i],avg_num_frames))
        dev_labels.append(cls)

train_all = np.array(train_all)
dev_all = np.array(dev_all)

train_extended = []
dev_extended = []
for i in range(len(train_all)):
    lst = []
    for j in range(len(train_all[i])):
        lst.extend(train_all[i][j])
    train_extended.append(np.array(lst))

for i in range(len(dev_all)):
    lst = []
    for j in range(len(dev_all[i])):
        lst.extend(dev_all[i][j])
    dev_extended.append(np.array(lst))


#Principal Component Analysis, for reducing Dimensionality
pca = PCA(0.99)
pca.fit(np.array(train_extended))
train_extended = pca.transform(train_extended)
dev_extended = pca.transform(dev_extended)

train_extended = np.array(train_extended)
dev_extended = np.array(dev_extended)

print(train_extended[0].shape)

def KNN_Isolated_digits(K):
    predictions = predict_KNN(train_extended,train_labels,dev_extended,K)
    count_correct = 0
    print(predictions)
    for i in range(len(predictions)):
        #print(predictions[i],dev_labels[i])
        if predictions[i] == dev_labels[i]:
            count_correct += 1
    print(count_correct/len(dev_labels))

KNN_Isolated_digits(5)
KNN_Isolated_digits(7)
KNN_Isolated_digits(9)
KNN_Isolated_digits(11)
KNN_Isolated_digits(14)