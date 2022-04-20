import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

#Reading Train and Dev Data
train_class_labels = []
train_data = []
with open("Synthetic_Dataset/train.txt") as f:
    temp = [[float(val) for val in line.strip().split(',')] for line in f]
    for i in temp:
        train_data.append([i[0],i[1]])
        train_class_labels.append(int(i[2]-1))

dev_class_labels = []
dev_data = []
with open("Synthetic_Dataset/dev.txt") as f:
    temp = [[float(val) for val in line.strip().split(',')] for line in f]
    for i in temp:
        dev_data.append([i[0],i[1]])
        dev_class_labels.append(int(i[2]-1))

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

clf = svm.SVC(C = 6e5,kernel='rbf')
clf.fit(train_data, train_class_labels)

predictions = clf.predict(dev_data)
count_correct = 0
for i in range(len(predictions)):
    if predictions[i] == dev_class_labels[i]:
        count_correct += 1

print(count_correct/len(dev_data))

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of SVC ')
# Set-up grid for plotting.
X0, X1 = np.array(dev_data)[: , 0], np.array(dev_data)[:,1]
xx, yy = make_meshgrid(X0,X1, 0.2)

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c= np.array(dev_class_labels), cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('y label here')
ax.set_xlabel('x label here')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()