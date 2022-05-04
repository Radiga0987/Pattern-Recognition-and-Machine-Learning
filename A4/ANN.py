from sklearn.neural_network import MLPClassifier
import numpy as np
import os
from scipy import signal
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import det_curve

#ROC and DET
def ROC_DET(S_list,class_labels,Title=""):
    temp = [] 

    # for score in S_list:
    #     sc_x = StandardScaler()
    #     score = sc_x.fit_transform(score)
    #     # for i in range(len(score[0])):
    #     #     sc_x = StandardScaler()
    #     #     score[:,i] = sc_x.fit_transform(score[:,i])
    #     temp.append(score)
    # S_list = temp
    #ROC
    Scores_list = []
    for case_no in range(len(S_list)):  #For Loop for all cases
        S = S_list[case_no]
        S = np.array(S)
        Scores = sorted(S.flatten())   #Scores are sorted for thresholding
        Scores_list.append(S.T.flatten())
        
        TPR = [0]*len(Scores)   
        FPR = [0]*len(Scores) 
        count=0
        for threshold in Scores:
            TP,FP,TN,FN = 0,0,0,0
            for i in range(len(S)):
                for j in range(len(S[0])):
                    if S[i][j] >= threshold:        #Classifying As Positive
                        if class_labels[i] == j+1: TP+=1
                        else:   FP+=1
                    else:
                        if class_labels[i] == j+1: FN+=1
                        else:   TN+=1
            TPR[count] = TP/(TP+FN)     #True Positive Rate
            FPR[count] = FP/(FP+TN)     #False Positive Rate

            count+=1
        plt.plot(FPR,TPR,label="Case "+str(case_no+1))
    plt.xlabel("False Postive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.title("Receiver Operating Characteristic curve "+Title)
    plt.show()

    #DET
    #Makes use of an inbuilt library - sklearn.metrics.det_curve
    #y_true - 1D array of length(no of class * len(dev_data))
    y_true = [0]*(len(S_list[0][0])*len(class_labels))
    count = 0
    for i in range(len(S[0])):
        for j in range(len(class_labels)):
            if class_labels[j] == i+1:
                y_true[count]=1
            count += 1
    plt.figure(figsize=(8,5))  

    #Plotting DET Curve for All Cases
    for i in range(len(S_list)):
        S = Scores_list[i]
        y_true = np.array(y_true)
        fpr, fnr, thresholds = det_curve(y_true, S)
        plt.plot(fpr,fnr,label="Case "+str(i+1))
        plt.yscale('logit')
        plt.xscale('logit')
    plt.legend()
    plt.xlabel("False Postive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("Detection Error Tradeoff curve "+Title)
    plt.show()

#Confusion Matrix
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


#########################
#Synthetic data

#Reading Train and Dev Data
train_class_labels = []
train_data = []
with open("Synthetic_Dataset/train.txt") as f:
    temp = [[float(val) for val in line.strip().split(',')] for line in f]
    for i in temp:
        train_data.append(np.array([i[0],i[1]]))
        train_class_labels.append(int(i[2]-1))
train_data = np.array(train_data)

class1,class2=[],[]
for i in range(len(train_data)):
    if train_class_labels[i]==0:
        class1.append(train_data[i])
    if train_class_labels[i]==1:
        class2.append(train_data[i])
class1=np.array(class1)
class2=np.array(class2)

dev_class_labels = []
dev_data = []
with open("Synthetic_Dataset/dev.txt") as f:
    temp = [[float(val) for val in line.strip().split(',')] for line in f]
    for i in temp:
        dev_data.append(np.array([i[0],i[1]]))
        dev_class_labels.append(int(i[2]-1))
dev_data = np.array(dev_data)


clf = MLPClassifier(solver='adam', alpha=0,hidden_layer_sizes=(10,10), random_state=1, max_iter = 2000)
clf.fit(train_data, train_class_labels)

predictions = clf.predict(dev_data)
scores = clf.predict_proba(dev_data)

conf_matrix = np.zeros((2,2))
count_correct = 0
for i in range(len(predictions)):
    if predictions[i] == dev_class_labels[i]:
        count_correct += 1
    conf_matrix[int(dev_class_labels[i])][int(predictions[i])] += 1
confusion_matrix(conf_matrix,"Synthetic Dataset - ANN")
    
print("Accuracy on dev data of synthetic dataset using ANN  =",count_correct/len(dev_class_labels))
ROC_DET([scores],np.array(dev_class_labels)+1,"Synthetic Dataset")

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

y_pred = 1+clf.predict(Space)
val_ = y_pred.reshape(500,500)

plt.figure(figsize=(7,7))
ax = plt.subplot()
ax.plot(class1[:,0],class1[:,1],'.',label="Class1  - yellow")
ax.plot(class2[:,0],class2[:,1],'.',label="Class2 - red")
c = ax.contourf(X, Y, val_,cmap='YlOrRd',levels=[0,1,2])
plt.show()


#########################
#Image Dataset

classes = ['coast','forest','highway','mountain','opencountry']

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


clf = MLPClassifier(solver='adam', activation="relu",alpha=2.7,hidden_layer_sizes=(128,64,32), random_state=1, max_iter = 5000)
clf.fit(train_imgs, train_img_label)

predictions = clf.predict(dev_imgs)
scores = clf.predict_proba(dev_imgs)

count_correct = 0
conf_matrix = np.zeros((5,5))
for i in range(len(predictions)):
    if predictions[i] == dev_img_label[i]:
        count_correct += 1
    conf_matrix[int(dev_img_label[i])][int(predictions[i])] += 1
confusion_matrix(conf_matrix,"Image Dataset - ANN")
    
print("Accuracy on dev data of Image dataset using ANN  =",count_correct/len(dev_img_label))
ROC_DET([scores],np.array(dev_img_label)+1,"Image Dataset")



#########################
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

for i in range(len(dev_all)):
    lst = []
    for j in range(len(dev_all[i])):
        lst.extend(dev_all[i][j])
    dev_extended.append(np.array(lst))


train_extended = np.array(train_extended)
dev_extended = np.array(dev_extended)


clf = MLPClassifier(solver='adam', activation="tanh",alpha=1,hidden_layer_sizes=(30,30), random_state=1, max_iter = 5000)
clf.fit(train_extended, train_labels)

predictions = clf.predict(dev_extended)
scores = clf.predict_proba(dev_extended)

count_correct = 0
conf_matrix = np.zeros((5,5))
for i in range(len(predictions)):
    if predictions[i] == dev_labels[i]:
        count_correct += 1
    conf_matrix[int(dev_labels[i])][int(predictions[i])] += 1
confusion_matrix(conf_matrix,"Isolated Digits Dataset ANN")

print("Accuracy on dev data of Isolated digits using ANN  =",count_correct/len(dev_labels))
ROC_DET([scores],np.array(dev_labels)+1,"Spoken Digits")

######################################

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

for i in range(len(dev_all)):
    lst = []
    for j in range(len(dev_all[i])):
        lst.extend(dev_all[i][j])
    dev_extended.append(np.array(lst))


train_extended = np.array(train_extended)
dev_extended = np.array(dev_extended)


clf = MLPClassifier(solver='adam', activation="tanh",alpha=2,hidden_layer_sizes=(100,50), random_state=1, max_iter = 5000)
clf.fit(train_extended, train_labels)

predictions = clf.predict(dev_extended)
scores = clf.predict_proba(dev_extended)

count_correct = 0
conf_matrix = np.zeros((5,5))
for i in range(len(predictions)):
    if predictions[i] == dev_labels[i]:
        count_correct += 1
    conf_matrix[int(dev_labels[i])][int(predictions[i])] += 1
confusion_matrix(conf_matrix,"Telugu Characters - ANN")

print("Accuracy on dev data of Telugu characters using ANN  =",count_correct/len(dev_labels))
ROC_DET([scores],np.array(dev_labels)+1,"Telugu Characters")