import numpy as np
import os
import time
import matplotlib.pyplot as plt

def Euclidean_Distance(v1,v2):
    d = np.linalg.norm(v1-v2)
    return(d)



def DTW(aud1,aud2):
    n,m = len(aud1),len(aud2)
    matrix = [[0 for j in range(m+1)] for i in range(n+1)]

    for i in range(n+1):
        matrix[i][0] = float("inf")

    for j in range(m+1):
        matrix[0][j] = float("inf")

    matrix[0][0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            matrix[i][j] = Euclidean_Distance(aud1[i-1],aud2[j-1]) + min([matrix[i-1][j], matrix[i][j-1], matrix[i-1][j-1]])

    return matrix[-1][-1]




def DTW_Classify(train,dev):
    predicted_classes=[]
    scores = []

    for test_cls in range(len(dev)):
        t=time.time()
        model_predictions = []
        for tst_sample in [dev[test_cls][0]]:
            min_dist_cls = []
            for trn_cls in range(len(train)):
                l=[]
                for trn_sample in [train[trn_cls][0]]:
                    l.append(DTW(trn_sample,tst_sample))
                min_dist_cls.append(min(l))
            scores.append(np.array(min_dist_cls))

            model_predictions.append(min_dist_cls.index(min(min_dist_cls)))
        predicted_classes.append(model_predictions)
        print("Classification of dev data of class",test_cls + 1,"completed")

    count_correct = 0
    count_dev_files = 0
    for i in range(len(predicted_classes)):
        for j in predicted_classes[i]:
            count_dev_files +=1
            if j == i:
                count_correct += 1
    Accuracy = count_correct/count_dev_files

    return predicted_classes,Accuracy,scores




def Confusion_matrix(predicted_classes,Title = "lol"):
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
        plt.plot(fpr,fnr,label="Case "+str(i+1))
        plt.yscale('logit')
        plt.xscale('logit')
    plt.legend()
    plt.xlabel("False Postive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("Detection Error Tradeoff curve")
    plt.show()




###########################################################################
digits = [1,2,5,9,'z']
train = []
dev = []

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
letters = ['a','bA','chA','lA','tA']
train = []
dev = []

for letter in letters:
    lst = []
    dir_list = os.listdir('Handwriting_Data//'+letter+'//train')
    for file in dir_list:
        temp = np.loadtxt('Handwriting_Data/'+letter+'/train/' + file)[1:]
        train_lst = np.array([temp[::2],temp[1::2]]).T
        train_lst = train_lst.tolist()
        for i in range(len(train_lst)-1):
            if train_lst[i+1][0] != train_lst[i][0]: 
                train_lst[i].append(np.arctan((train_lst[i+1][1] - train_lst[i][1])/(train_lst[i+1][0] - train_lst[i][0])))
            else:
                if train_lst[i+1][1] - train_lst[i][1] > 0:
                    train_lst[i].append(np.arctan(np.inf))
                else:
                    train_lst[i].append(np.arctan(-np.inf))
        train_lst[len(train_lst) - 1].append(train_lst[len(train_lst) - 2][2])

        lst.append(np.array(train_lst))

    train.append(lst)

    lst = []
    dir_list = os.listdir('Handwriting_Data//'+letter+'//dev')
    for file in dir_list:
        temp = np.loadtxt('Handwriting_Data/'+letter+'/dev/' + file)[1:]
        test_lst = np.array([temp[::2],temp[1::2]]).T
        test_lst = test_lst.tolist()
        for i in range(len(test_lst)-1):
            if test_lst[i+1][0] != test_lst[i][0]: 
                test_lst[i].append(np.arctan((test_lst[i+1][1] - test_lst[i][1])/(test_lst[i+1][0] - test_lst[i][0])))
            else:
                if test_lst[i+1][1] - test_lst[i][1] > 0:
                    test_lst[i].append(np.arctan(np.inf))
                else:
                    test_lst[i].append(np.arctan(-np.inf))
        test_lst[len(test_lst) - 1].append(test_lst[len(test_lst) - 2][2])

        lst.append(np.array(test_lst))

    dev.append(lst)


t=time.time()
print("Classifying all dev data for Handwriting Data")
predicted_classes,Accuracy,scores = DTW_Classify(train,dev)
print(predicted_classes)
Confusion_matrix(predicted_classes)
ROC_DET(np.array(scores),[1]*len(dev[0])+[2]*len(dev[1])+[3]*len(dev[2])+[4]*len(dev[3])+[5]*len(dev[4]),"DTW for letters")
print(time.time()-t)