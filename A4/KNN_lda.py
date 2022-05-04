import numpy as np
import os
import time
from scipy import signal
import math
from pca_lda_ import pca_,lda_,transform

import warnings
warnings.filterwarnings("ignore")
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

def KNN_Synthetic(K,train_data,train_labels,dev_data,dev_labels):
    predictions = predict_KNN(train_data,train_labels,dev_data,K)

    count_correct = 0
    for i in range(len(predictions)):
        if predictions[i] == dev_labels[i]:
            count_correct += 1
    acc = count_correct/len(dev_data)
    print("Accuracy on dev data of synthetic dataset using KNN  =",acc * 100,"%")
    return acc

def KNN_Images(K,train_imgs,train_labels,dev_imgs,dev_labels):
    predictions = predict_KNN(train_imgs,train_labels,dev_imgs,K)

    count_correct = 0
    for i in range(len(predictions)):
        if predictions[i] == dev_labels[i]:
            count_correct += 1
    acc = count_correct/len(dev_labels)
    print("Accuracy on dev data of image dataset using KNN  =",acc * 100,"%")
    return acc

def KNN_Isolated_digits(K,train_ext,train_labels,dev_ext,dev_labels):
    predictions = predict_KNN(train_ext,train_labels,dev_ext,K)

    count_correct = 0
    for i in range(len(predictions)):
        if predictions[i] == dev_labels[i]:
            count_correct += 1
    acc = count_correct/len(dev_labels)
    print("Accuracy on dev data of isolated digits dataset using KNN  =",acc * 100,"%")
    return acc

def KNN_Telugu_chars(K,train_ext,train_labels,dev_ext,dev_labels):
    predictions = predict_KNN(train_ext,train_labels,dev_ext,K)

    count_correct = 0
    for i in range(len(predictions)):
        if predictions[i] == dev_labels[i]:
            count_correct += 1
    acc = count_correct/len(dev_labels)
    print("Accuracy on dev data of Telugu characters using KNN  =",acc * 100,"%")
    return acc

##########################################
# Synthetic data
# Reading Train and Dev Data
train_class_labels = []
train_data = []
with open("Synthetic_Dataset/train.txt") as f:
    temp = [[float(val) for val in line.strip().split(',')] for line in f]
    for i in temp:
        train_data.append(np.array([i[0],i[1]]))
        train_class_labels.append(int(i[2]-1))
train_data = np.array(train_data)

dev_class_labels = []
dev_data = []
with open("Synthetic_Dataset/dev.txt") as f:
    temp = [[float(val) for val in line.strip().split(',')] for line in f]
    for i in temp:
        dev_data.append(np.array([i[0],i[1]]))
        dev_class_labels.append(int(i[2]-1))
dev_data = np.array(dev_data)

KNN_Synthetic(3,train_data,train_class_labels,dev_data,dev_class_labels)

############################################
# Image dataset

classes = ['coast','forest','highway','mountain','opencountry']
denoms = []
train_imgs = []
dev_imgs = []
dev_img_label= []
train_img_label= []


#Reading the Images of each class for train and Developement DataSets
for i,cls in enumerate(classes):
    dir_list = os.listdir('Features//'+cls+'//train')
    for file in dir_list:
        concat = np.array([])
        for r in np.loadtxt('Features/'+cls+'/train/' + file):
            concat = np.concatenate((concat,r))
        train_imgs.append(concat)
        train_img_label.extend([i])

    dir_list = os.listdir('Features//'+cls+'//dev')
    for file in dir_list:
        concat = np.array([])
        for r in np.loadtxt('Features/'+cls+'/dev/' + file):
            concat = np.concatenate((concat,r))
        dev_imgs.append(concat)
        dev_img_label.extend([i])

#Mean Normalisation
mean_train = np.mean(train_imgs,axis=0)
maxs = np.max(train_imgs,axis=0)
mins = np.min(train_imgs,axis=0)
denoms = maxs - mins
train_imgs = (train_imgs-mean_train)/denoms
dev_imgs = (dev_imgs-mean_train)/denoms

train_imgs = np.array(train_imgs)
dev_imgs = np.array(dev_imgs)

Q = lda_(train_imgs,train_img_label,.99)
train_imgs = transform(Q,train_imgs)
dev_imgs = transform(Q,dev_imgs)

KNN_Images(10,train_imgs,train_img_label,dev_imgs,dev_img_label)





############################################
# Isolated digits
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

train_extended = np.array(train_extended)
dev_extended = np.array(dev_extended)

mean_train = np.mean(train_extended,axis=0)
maxs = np.max(train_extended,axis=0)
mins = np.min(train_extended,axis=0)
denoms = maxs - mins
train_extended = (train_extended-mean_train)/denoms
dev_extended = (dev_extended-mean_train)/denoms

Q = lda_(train_extended,train_labels,.99)
train_extended = transform(Q,train_extended)
dev_extended = transform(Q,dev_extended)

KNN_Isolated_digits(7,train_extended,train_labels,dev_extended,dev_labels)


############################################
# Handwriting Data
letters = ['a','bA','chA','lA','tA']
train = []
dev = []

#Loading train and dev data
for letter in letters:
    lst = []
    dir_list = os.listdir('Handwriting_Data//'+letter+'//train')
    for file in dir_list:
        temp = np.loadtxt('Handwriting_Data/'+letter+'/train/' + file)[1:]
        train_lst = np.array([temp[::2],temp[1::2]]).T
        train_lst = train_lst.tolist()
        #Here we add an extra feature as mentioned in the report.This feature is arctan(slope) which is basically the angle made by tangent with x axis
        #For reason we add this angle in radians and not the slope directly, please refer to the report
        for i in range(len(train_lst)-1): 
            if train_lst[i+1][0] != train_lst[i][0]: 
                train_lst[i].append(np.arctan((train_lst[i+1][1] - train_lst[i][1])/(train_lst[i+1][0] - train_lst[i][0])))
            else: #If x coordinates of consecutive points are same, then we cannot divide by x2-x1 and hence directly take arctan on +-np.inf
                if train_lst[i+1][1] - train_lst[i][1] > 0: #arctan(np.inf) if y2 - y1 > 0
                    train_lst[i].append(np.arctan(np.inf))
                else:
                    train_lst[i].append(np.arctan(-np.inf)) #arctan(-np.inf) if y1 - y2 > 0
        train_lst[len(train_lst) - 1].append(train_lst[len(train_lst) - 2][2])

        lst.append(np.array(train_lst))

    train.append(lst)

    lst = []
    dir_list = os.listdir('Handwriting_Data//'+letter+'//dev')
    for file in dir_list:
        temp = np.loadtxt('Handwriting_Data/'+letter+'/dev/' + file)[1:]
        test_lst = np.array([temp[::2],temp[1::2]]).T
        test_lst = test_lst.tolist()
        for i in range(len(test_lst)-1):# The same extra feature of angle is added to the dev data too
            if test_lst[i+1][0] != test_lst[i][0]: 
                test_lst[i].append(np.arctan((test_lst[i+1][1] - test_lst[i][1])/(test_lst[i+1][0] - test_lst[i][0])))
            else: #If x coordinates of consecutive points are same, then we cannot divide by x2-x1 and hence directly take arctan on +-np.inf
                if test_lst[i+1][1] - test_lst[i][1] > 0: #arctan(np.inf) if y2 - y1 > 0
                    test_lst[i].append(np.arctan(np.inf))
                else:
                    test_lst[i].append(np.arctan(-np.inf)) #arctan(-np.inf) if y1 - y2 > 0
        test_lst[len(test_lst) - 1].append(test_lst[len(test_lst) - 2][2])

        lst.append(np.array(test_lst))

    dev.append(lst)

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

train_extended = np.array(train_extended)
dev_extended = np.array(dev_extended)

mean_train = np.mean(train_extended,axis=0)
maxs = np.max(train_extended,axis=0)
mins = np.min(train_extended,axis=0)
denoms = maxs - mins
train_extended = (train_extended-mean_train)/denoms
dev_extended = (dev_extended-mean_train)/denoms

Q = lda_(train_extended,train_labels,.99)
train_extended = transform(Q,train_extended)
dev_extended = transform(Q,dev_extended)

KNN_Telugu_chars(20,train_extended,train_labels,dev_extended,dev_labels)