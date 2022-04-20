#Importing required libraries
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from numba import njit

def Euclidean_Distance(v1,v2):
    d = np.linalg.norm(v1-v2)
    return(d)

"""IMPORTANT
Please note that we have used a decorator here from Numba 
called njit.Numba is a just-in-time compiler for Python that 
works best on code that uses NumPy arrays and functions, and loops.
This converts the code into machine efficient code hence improving 
runtime drastically.
Before running this code, one needs to
pip install numba (or) conda install numba 

Else, comment out the import of njit in line 6 and line 24 were @njit is used
but this will result in longer runtime by around 15 mins for each of the datasets
"""
@njit 
def DTW(t1,t2):     #Function that performs DTW 
    n,m = len(t1),len(t2)
    matrix = np.zeros((n+1,m+1))   #Table to store distances 

    for i in range(n+1): #First row has value as infinite
        matrix[i,0] = np.inf

    for j in range(m+1): #First coloumn has value as infinite
        matrix[0,j] = np.inf

    matrix[0,0] = 0   # First value of the table is set to 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            matrix[i,j] = np.sum((t1[i-1]-t2[j-1])**2) + min([matrix[i-1][j], matrix[i][j-1], matrix[i-1][j-1]])  #Minimizing each step

    return matrix[-1,-1]



#Function that performs DTW on each dev sample on all train samples and then classifies
def DTW_Classify(train,dev):
    predicted_classes=[]
    scores = []

    #We iterate over all the dev samples
    for test_cls in range(len(dev)):
        t=time.time()
        model_predictions = []
        for tst_sample in dev[test_cls]:
            min_dist_cls = []
            for trn_cls in range(len(train)):#Now for a given dev sample, we iterate over all train samples
                l=[]
                for trn_sample in train[trn_cls]:
                    l.append(DTW(trn_sample,tst_sample)) #DTW is performed and the distances are found
                min_dist_cls.append(min(l))  #For each dev sample we obtain the best distance values corresponding to each class 
            scores.append(np.array(min_dist_cls))

            model_predictions.append(min_dist_cls.index(min(min_dist_cls))) #We classify the dev samples by comparing these distances for different classes
        predicted_classes.append(model_predictions)
        print("Classification of dev data of class",test_cls + 1,"completed")

    # Finding percentage of dev samples correctly classified (Accuracy)
    count_correct = 0
    count_dev_files = 0
    for i in range(len(predicted_classes)):
        for j in predicted_classes[i]:
            count_dev_files +=1
            if j == i:
                count_correct += 1
    Accuracy = (count_correct/count_dev_files)*100

    return predicted_classes,Accuracy,scores



#Function for plotting confusion matrix
def Confusion_matrix(predicted_classes,Title = "Confusion Matrix"):
    conf_matrix = np.zeros((len(predicted_classes),len(predicted_classes)))
    for cls in range(len(predicted_classes)):
        for val in predicted_classes[cls]:
            conf_matrix[cls][val] += 1

    #Plotting the Confusion Matrix
    fig, ax = plt.subplots(figsize=(5,5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.6)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='large')
    plt.xlabel('Predictions', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title(Title, fontsize=12)
    plt.show()


#Function for obtaining ROC and DET curves
from sklearn.metrics import det_curve
def ROC_DET(S,ground_truth,label ="DTW ROC curve"):
    #ROC
    S_list = []
    plt.figure(figsize=(6,5))
    for case_no in range(1,2):  #For Loop for all cases
        Scores = sorted(S.flatten())   #Scores are sorted for thresholding
        S_list.append(S.flatten())

        TPR = [0]*len(Scores)   
        FPR = [0]*len(Scores) 
        count=0
        for threshold in Scores:
            TP,FP,TN,FN = 0,0,0,0
            for i in range(len(ground_truth)):
                for j in range(3):
                    if S[i][j] <= threshold:        #Classifying As Positive
                        if ground_truth[i] == j+1: TP+=1
                        else:   FP+=1
                    else:
                        if ground_truth[i] == j+1: FN+=1
                        else:   TN+=1
            TPR[count] = TP/(TP+FN)     #True Positive Rate
            FPR[count] = FP/(FP+TN)     #False Positive Rate

            count+=1
        plt.plot(FPR,TPR,label=label)
    plt.xlabel("False Postive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic curve")
    plt.legend()
    plt.show()

    #DET
    #Makes use of an inbuilt library - sklearn.metrics.det_curve
    #y_true - 1D array of length(no of class * len(dev_data))
    y_true = []
    for i in range(5):
        for j in range(len(ground_truth)):
            if ground_truth[j] == i+1:
                y_true.append(1)
            else:
                y_true.append(0)

    plt.figure(figsize=(6,5))    
    for i in range(1):
        S = S_list[i]
        y_true = np.array(y_true)
        fpr, fnr, thresholds = det_curve(y_true, S)
        plt.plot(fpr,fnr,label=label)
        plt.plot(fpr,fnr,label="Case "+str(i+1))
        plt.yscale('logit')
        plt.xscale('logit')
    plt.legend()
    plt.xlabel("False Postive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("Detection Error Tradeoff curve")
    plt.show()




###########################################################################

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
ground_truth = [1]*len(dev[0])+[2]*len(dev[1])+[3]*len(dev[2])+[4]*len(dev[3])+[5]*len(dev[4])
t=time.time()

#Classifying all the dev data using train data
print("Classifying all dev data for Isolated Digits")
predicted_classes,Accuracy,scores = DTW_Classify(train,dev)
print(time.time()-t)
print("Accuracy (Isolated digits) = ", Accuracy,"%")
print("The class predictions by DTW for Isolated digits is ",predicted_classes)
Confusion_matrix(predicted_classes,"DTW on Digits")
ROC_DET(np.array(scores),ground_truth,"DTW on Digits")

for digit in digits:
    lst = []
    dir_list = os.listdir('Isolated_Digits//'+str(digit)+'//train')
    for file in dir_list:
        if file.endswith('.mfcc'):
            lst.append(np.loadtxt('Isolated_Digits/'+str(digit)+'/train/' + file,skiprows = 1))

    train.append(lst)

    lst = []
    dir_list = os.listdir('Isolated_Digits//'+str(digit)+'//dev')
    for file in dir_list:
        if file.endswith('.mfcc'):
            lst.append(np.loadtxt('Isolated_Digits/'+str(digit)+'/dev/' + file,skiprows = 1))

    dev.append(lst)
t=time.time()
print("Classifying all dev data for Isolated Digits")
predicted_classes,Accuracy,scores = DTW_Classify(train,dev)
Confusion_matrix(predicted_classes)
ROC_DET(np.array(scores),[1,2,3,4,5])#[1]*len(dev[0])+[2]*len(dev[1])+[3]*len(dev[2])+[4]*len(dev[3])+[5]*len(dev[4]),"DTW for digits")
print(time.time()-t)

###########################################################################

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
ground_truth = [1]*len(dev[0])+[2]*len(dev[1])+[3]*len(dev[2])+[4]*len(dev[3])+[5]*len(dev[4])

#Classifying all the dev data using train data
t=time.time()
print("Classifying all dev data for Handwriting Data")
predicted_classes,Accuracy,scores = DTW_Classify(train,dev)
print(time.time()-t)
print("Accuracy (Telugu characters) = ",Accuracy,"%")
print("The class predictions by DTW for Telugu characters is ",predicted_classes)
Confusion_matrix(predicted_classes,"DTW on Telugu letters")
ROC_DET(np.array(scores),ground_truth,"DTW on Telugu letters")
