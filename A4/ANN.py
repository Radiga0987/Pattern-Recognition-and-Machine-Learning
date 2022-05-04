from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import time
from sklearn.decomposition import PCA

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

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(train_data, train_class_labels)