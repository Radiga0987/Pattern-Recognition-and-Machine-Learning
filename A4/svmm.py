import numpy as np
from sklearn import svm
from sklearn.metrics import det_curve
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from scipy import signal
import math


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

#ROC and DET
def ROC_DET(S_list,class_labels,Title=""):
    temp = [] 

    for score in S_list:
        sc_x = StandardScaler()
        score = sc_x.fit_transform(score)
        # for i in range(len(score[0])):
        #     sc_x = StandardScaler()
        #     score[:,i] = sc_x.fit_transform(score[:,i])
        temp.append(score)
    S_list = temp
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

#Synthetic Dataset
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

    #Train
    clf = svm.SVC(C = 6e5,kernel='rbf',probability=True)
    clf.fit(train_data[:,:2], train_data[:,2])

    #Classify
    y_pred = clf.predict(dev_data[:,:2])
    scores = clf.predict_proba(dev_data[:,:2])
    print("Accuracy of Classification",(1-sum(abs(y_pred-dev_data[:,2]))/len(dev_data))*100,"%")
    
    #Conf and ROC and DET
    conf_matrix = np.zeros((no_classes,no_classes))
    for i in range(len(dev_data)):
        conf_matrix[int(dev_data[:,2][i]-1)][int(y_pred[i]-1)] += 1
    confusion_matrix(conf_matrix,"Synthetic Dataset Logistic Regression")
    
    ROC_DET([scores],dev_data[:,2],"Synthetic")

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

    y_pred = clf.predict(Space)
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
    train_all = []
    train_labels = []
    dev_labels= []
    dev = []

    #Reading the Images of each class for train and Developement DataSets
    for i,cls in enumerate(classes):
        dir_list = os.listdir('Features//'+cls+'//train')
        for file in dir_list:
            train_labels.append(i+1)            
            image =  np.loadtxt('Features/'+cls+'/train/' + file)     
            train_all.append(image.flatten())
      
        dir_list = os.listdir('Features//'+cls+'//dev')
        for file in dir_list:
            dev_labels.append(i+1)
            image =  np.loadtxt('Features/'+cls+'/dev/' + file)
            dev.append(image.flatten())
    
    train_all = np.array(train_all)
    train_labels = np.array(train_labels)
    
    #Normalise Data
    mean_train = np.mean(train_all,axis=0)
    maxs = np.max(train_all,axis=0)
    mins = np.min(train_all,axis=0)
    denoms = maxs - mins

    train_all = (train_all-mean_train)/denoms
    dev = (dev-mean_train)/denoms

    #Train
    clf = svm.SVC(C = 6e5,kernel='rbf',probability=True)
    clf.fit(train_all, train_labels)
    
    #predict
    y_pred = clf.predict(dev)
    scores = clf.predict_proba(dev)
    y_pred = np.array(y_pred)
    
    print("Accuracy on dev data of image dataset using SVM",(1-np.count_nonzero(y_pred-dev_labels)/len(dev_labels))*100,"%")

    #Conf matrix and Roc and det
    conf_matrix = np.zeros((no_classes,no_classes))
    for i in range(len(dev_labels)):
        conf_matrix[int(dev_labels[i]-1)][int(y_pred[i]-1)] += 1
    confusion_matrix(conf_matrix,"Image Dataset Logistic Regression")

    ROC_DET([scores],dev_labels,"Image")

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

    
    #Finding avg no:of frames
    count = 0
    num_frames = 0
    for cls in range(len(train)):
        for i in range(len(train[cls])):
            num_frames += len(train[cls][i])
            count += 1
    avg_num_frames = math.floor(num_frames/count)


    #Resampling
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

    #Mean Normalise
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

    #Train
    clf = svm.SVC(C = 6e5,kernel='rbf',probability=True)
    clf.fit(train_extended, train_labels)

    #Predict
    y_pred = clf.predict(dev_extended)
    scores = clf.predict_proba(dev_extended)
    y_pred = np.array(y_pred)
    
    print("Accuracy on dev data of Isolated Digits using SVM",(1-np.count_nonzero(y_pred-dev_labels)/len(dev_labels))*100,"%")

    conf_matrix = np.zeros((no_classes,no_classes))
    for i in range(len(dev_labels)):
        conf_matrix[int(dev_labels[i]-1)][int(y_pred[i]-1)] += 1
    confusion_matrix(conf_matrix,"Spoken Digits Dataset Logistic Regression")

    ROC_DET([scores],dev_labels,"Spoken digits")
    
def Handwriting_Data():
    # Handwriting Data
    letters = ['a','bA','chA','lA','tA']
    no_classes = 5
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

    #Avg no: of samples
    count = 0
    num_frames = 0
    for cls in range(len(train)):
        for i in range(len(train[cls])):
            num_frames += len(train[cls][i])
            count += 1
    avg_num_frames = math.floor(num_frames/count)

    #Resample
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
    dev_all = np.array(dev_all)

    #Mean Normalise
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

    train_labels = np.array(train_labels)
    train_extended = np.array(train_extended)
    dev_extended = np.array(dev_extended)

    #Train
    clf = svm.SVC(C = 6e5,kernel='rbf',probability=True)
    clf.fit(train_extended, train_labels)

    #Predict
    y_pred = clf.predict(dev_extended)
    scores = clf.predict_proba(dev_extended)
    y_pred = np.array(y_pred)
    
    print("Accuracy on dev data of Telugu characters using SVM",(1-np.count_nonzero(y_pred-dev_labels)/len(dev_labels))*100,"%")

    conf_matrix = np.zeros((no_classes,no_classes))
    for i in range(len(dev_labels)):
        conf_matrix[int(dev_labels[i]-1)][int(y_pred[i]-1)] += 1
    confusion_matrix(conf_matrix,"Handwritten Data Dataset Logistic Regression")

    ROC_DET([scores],dev_labels,"Handwritten")


Synth()
Image_Dataset()
spoken_digits()
Handwriting_Data()