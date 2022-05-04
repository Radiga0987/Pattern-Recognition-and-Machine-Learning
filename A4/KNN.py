import numpy as np
import os
import time
from sklearn.decomposition import PCA
from scipy import signal
import math

#Reading Train and Dev Data
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

def KNN_Synthetic():
    t1 = time.time()
    predictions = predict_KNN(train_data,train_class_labels,dev_data,3)
    print(time.time()-t1)
    count_correct = 0
    for i in range(len(predictions)):
        if predictions[i] == dev_class_labels[i]:
            count_correct += 1
    print(count_correct/len(dev_data))

#KNN_Synthetic()



classes = ['coast','forest','highway','mountain','opencountry']
train = []  
train_all = []
dev_all = []
dev = []
test = []
denoms = []
priors = np.zeros(len(classes))
train_labels = []
dev_labels = []
dev_img_label= []
train_img_label= []
#Reading the Images of each class for train and Developement DataSets
for i,cls in enumerate(classes):
    lst = []
    dir_list = os.listdir('Features//'+cls+'//train')
    priors[i] = len(dir_list)
    for file in dir_list:
        lst.extend(np.loadtxt('Features/'+cls+'/train/' + file))
        train_labels.extend([i]*36)
        train_img_label.extend([i])
    lst = np.array(lst)
    train.append(lst)
    train_all.extend(lst)

    lst = []
    lst2 = []
    dir_list = os.listdir('Features//'+cls+'//dev')
    for file in dir_list:
        lst.append(np.loadtxt('Features/'+cls+'/dev/' + file))
        lst2.extend(np.loadtxt('Features/'+cls+'/dev/' + file))
        dev_labels.extend([i]*36)
        dev_img_label.extend([i])
    dev.append(np.array(lst))
    test.append(np.array(lst2))
    dev_all.extend(lst2)
# train_all = np.array(train_all)
# dev_all = np.array(dev_all)
#Mean Normalisation
mean_train = np.mean(train_all,axis=0)
maxs = np.max(train_all,axis=0)
mins = np.min(train_all,axis=0)
denoms = maxs - mins
train_all = (train_all-mean_train)/denoms
dev_all = (dev_all-mean_train)/denoms



#Principal Component Analysis, for reducing Dimensionality
pca = PCA(.99)
pca.fit(np.array(train_all))
train_all = pca.transform(train_all)
dev_all = pca.transform(dev_all)

train_all = np.array(train_all)
dev_all = np.array(dev_all)

def KNN_Images():
    t1 = time.time()
    predictions = predict_KNN(train_all,train_labels,dev_all,18)
    #print(predictions)
    print(time.time()-t1)
    count_correct = 0
    for i in range(len(predictions)):
        #print(predictions[i],dev_labels[i])
        if predictions[i] == dev_labels[i]:
            count_correct += 1
    #print(count_correct/len(dev_all))

    predictions_img = []
    for i in range(len(dev_img_label)):
        predictions_img.append(np.bincount(predictions[36*i:36*(i+1)]).argmax())
    predictions_img = np.array(predictions_img)
    print(predictions_img)

    count_correct = 0
    for i in range(len(predictions_img)):
        if predictions_img[i] == dev_img_label[i]:
            count_correct += 1
    print(count_correct/len(dev_img_label))

#KNN_Images()

#########################################


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

def KNN_Isolated_digits():
    t1 = time.time()
    predictions = predict_KNN(train_all,train_labels,dev_all,18)
    #print(predictions)
    print(time.time()-t1)
    count_correct = 0
    for i in range(len(predictions)):
        #print(predictions[i],dev_labels[i])
        if predictions[i] == dev_labels[i]:
            count_correct += 1
    #print(count_correct/len(dev_all))

    predictions_img = []
    for i in range(len(dev_img_label)):
        predictions_img.append(np.bincount(predictions[36*i:36*(i+1)]).argmax())
    predictions_img = np.array(predictions_img)
    print(predictions_img)

    count_correct = 0
    for i in range(len(predictions_img)):
        if predictions_img[i] == dev_img_label[i]:
            count_correct += 1
    print(count_correct/len(dev_img_label))

KNN_Isolated_digits()